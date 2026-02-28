import numpy as np
import torch
import os
import cv2  # 建议移到顶部
from .base import BasePerceptor, PerceptionResult
from typing import Optional, Dict

# --- 关键修复：导入 pdi_logger ---
from ..utils.logger import pdi_logger 

# 尝试导入，如果失败则在 infer 时报错
try:
    from cotracker.predictor import CoTrackerPredictor
except ImportError:
    CoTrackerPredictor = None

class TrackWrapper(BasePerceptor):
    """基于 Co-Tracker v3 的微观运动监控封装器"""
    
    def __init__(self, checkpoint: Optional[str] = None, device: str = "cuda"):
        super().__init__(device)
        self.model = self._load_model(checkpoint)

    def _load_model(self, checkpoint):
        """
        处理版本不匹配的核心逻辑
        """
        pdi_logger.info(f"正在初始化 Co-Tracker (设备: {self.device})...")
        
        # 如果没有指定路径或路径无效，尝试从 torch.hub 加载
        if checkpoint is None or not os.path.exists(checkpoint):
            pdi_logger.warning("未发现有效本地权重，尝试通过 torch.hub 加载...")
            # 修正官方 v3 调用名为 cotracker3_offline
            return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)

        try:
            model = CoTrackerPredictor(checkpoint=checkpoint).to(self.device)
            pdi_logger.success(f"成功加载本地权重: {checkpoint}")
            return model
        except RuntimeError as e:
            pdi_logger.warning(f"本地权重加载失败: {str(e)[:50]}...")
            pdi_logger.info("正在自动切换至 torch.hub 下载匹配模型...")
            return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(self.device)

    def infer(
        self,
        video_path: str,
        initial_mask: np.ndarray,
        grid_size: int = 10,
        bg_grid_size: int = 15,
        **kwargs,
    ) -> PerceptionResult:
        """前景+背景一次性追踪。

        背景点通过在 mask 外区域均匀采样获得，与前景点合并后送入同一次
        Co-Tracker 推理，推理结束后再按 n_fg 拆分，避免重复加载模型。
        bg_tracks 存入 metadata['bg_tracks'] / metadata['bg_visibility']。
        """
        import cv2

        # 1. 读取视频并进行空间缩放
        cap = cv2.VideoCapture(video_path)
        frames = []
        max_dim = 880
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        orig_h, orig_w = initial_mask.shape
        curr_h, curr_w = frames[0].shape[:2]
        scale_x, scale_y = orig_w / curr_w, orig_h / curr_h

        video_np = np.stack(frames)
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].to(self.device)

        # 缩放 mask
        small_mask = cv2.resize(
            initial_mask.astype(np.uint8), (curr_w, curr_h),
            interpolation=cv2.INTER_NEAREST,
        )
        small_mask = (small_mask > 0).astype(np.uint8)

        # --- 2. 前景 queries ---
        yy_fg, xx_fg = np.where(small_mask > 0)
        if len(yy_fg) == 0:
            pdi_logger.warning("初始 mask 无前景像素，仅使用背景网格追踪")
            fg_queries_np = np.empty((0, 3), dtype=np.float32)
        else:
            n_fg = min(grid_size * grid_size, len(yy_fg))
            idx_fg = np.linspace(0, len(yy_fg) - 1, n_fg, dtype=int)
            fg_queries_np = np.stack(
                [np.zeros(n_fg), xx_fg[idx_fg].astype(np.float32), yy_fg[idx_fg].astype(np.float32)],
                axis=1,
            )

        # --- 3. 背景 queries ---
        yy_bg, xx_bg = np.where(small_mask == 0)
        if len(yy_bg) > 0:
            n_bg = min(bg_grid_size * bg_grid_size, len(yy_bg))
            idx_bg = np.linspace(0, len(yy_bg) - 1, n_bg, dtype=int)
            bg_queries_np = np.stack(
                [np.zeros(n_bg), xx_bg[idx_bg].astype(np.float32), yy_bg[idx_bg].astype(np.float32)],
                axis=1,
            )
        else:
            bg_queries_np = np.empty((0, 3), dtype=np.float32)

        # --- 4. 合并 queries 送入推理 ---
        n_fg_pts = len(fg_queries_np)
        all_queries_np = np.vstack([fg_queries_np, bg_queries_np]).astype(np.float32)

        pdi_logger.info(
            f"Co-Tracker 追踪 (尺寸:{curr_w}x{curr_h}, "
            f"前景:{n_fg_pts}点, 背景:{len(bg_queries_np)}点)..."
        )

        if len(all_queries_np) >= 2:
            queries = torch.from_numpy(all_queries_np[None]).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        queries=queries,
                        grid_size=0,
                        grid_query_frame=0,
                    )
        else:
            # 兜底：全图网格
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        grid_size=grid_size,
                        grid_query_frame=0,
                    )
            n_fg_pts = tracks.shape[2]  # 全部算前景

        # --- 5. 坐标还原 & 前背景拆分 ---
        tracks_np = tracks[0].cpu().numpy()   # (T, N_total, 2)
        tracks_np[:, :, 0] *= scale_x
        tracks_np[:, :, 1] *= scale_y
        vis_np = visibility[0].cpu().numpy()  # (T, N_total)

        fg_tracks = tracks_np[:, :n_fg_pts, :]
        bg_tracks = tracks_np[:, n_fg_pts:, :]
        fg_vis    = vis_np[:, :n_fg_pts]
        bg_vis    = vis_np[:, n_fg_pts:]

        breathing_metric = self.calculate_breathing_artifact(fg_tracks)

        del video_tensor
        torch.cuda.empty_cache()

        return PerceptionResult(
            video_id=os.path.basename(video_path),
            frames_count=len(fg_tracks),
            masks=np.zeros((1, 1, 1)),
            h_pixel=np.zeros(len(fg_tracks)),
            x_center=np.zeros(len(fg_tracks)),
            tracks_2d=fg_tracks,
            confidence=fg_vis,
            metadata={
                "breathing_metric": breathing_metric,
                "bg_tracks": bg_tracks,      # (T, N_bg, 2) — 背景轨迹
                "bg_visibility": bg_vis,     # (T, N_bg)
            },
        )

    def calculate_breathing_artifact(self, tracks: np.ndarray) -> float:
        """
        计算点云内部的相对距离波动 (Coefficient of Variation)
        """
        T, N, _ = tracks.shape
        if N < 2: return 0.0
        
        group_a = tracks[:, :min(5, N//2), :]
        group_b = tracks[:, -min(5, N//2):, :]
        
        centroid_a = np.mean(group_a, axis=1)
        centroid_b = np.mean(group_b, axis=1)
        
        dists = np.linalg.norm(centroid_a - centroid_b, axis=1)
        
        if np.mean(dists) < 1e-6: return 0.0
        return float(np.std(dists) / np.mean(dists))