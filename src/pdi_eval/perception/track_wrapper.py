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

        # --- 2. 前景 queries（空间均匀网格采样）---
        fg_queries_np = self._grid_sample_queries(small_mask, region=1, n=grid_size * grid_size)

        # --- 3. 背景 queries ---
        bg_queries_np = self._grid_sample_queries(small_mask, region=0, n=bg_grid_size * bg_grid_size)

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

        # --- 6. 追踪质量过滤 ---
        fg_tracks, fg_vis = self._filter_tracks(fg_tracks, fg_vis)
        bg_tracks, bg_vis = self._filter_tracks(bg_tracks, bg_vis)

        breathing_metric = self.calculate_breathing_artifact(fg_tracks)

        del video_tensor
        torch.cuda.empty_cache()

        n_fg_kept = fg_tracks.shape[1]
        tracking_confidence = float(fg_vis.mean()) if fg_vis.size > 0 else 0.0
        pdi_logger.info(f"追踪完成: 前景保留 {n_fg_kept} 点, 平均可见度 {tracking_confidence:.3f}")

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
                "tracking_confidence": tracking_confidence,
                "bg_tracks": bg_tracks,
                "bg_visibility": bg_vis,
            },
        )

    def _grid_sample_queries(
        self,
        mask: np.ndarray,
        region: int,
        n: int,
    ) -> np.ndarray:
        """在 mask 指定区域（region=1 前景 / region=0 背景）做空间均匀网格采样。

        将区域划分为 sqrt(n) x sqrt(n) 的子格，每格随机取一点，避免点簇聚。
        返回 (M, 3) 的 queries，格式为 [frame=0, x, y]，M <= n。
        """
        yy, xx = np.where(mask == region)
        if len(yy) == 0:
            if region == 1:
                pdi_logger.warning("初始 mask 无前景像素，仅使用背景网格追踪")
            return np.empty((0, 3), dtype=np.float32)

        n = min(n, len(yy))
        side = max(1, int(np.ceil(np.sqrt(n))))
        h, w = mask.shape
        cell_h = max(1, h // side)
        cell_w = max(1, w // side)

        rng = np.random.default_rng(42)
        pts = []
        for gy in range(side):
            for gx in range(side):
                y0, y1 = gy * cell_h, min((gy + 1) * cell_h, h)
                x0, x1 = gx * cell_w, min((gx + 1) * cell_w, w)
                in_cell = np.where((yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1))[0]
                if len(in_cell) > 0:
                    pick = rng.choice(in_cell)
                    pts.append([0.0, float(xx[pick]), float(yy[pick])])
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break

        if not pts:
            # fallback: 随机采样
            idx = rng.choice(len(yy), min(n, len(yy)), replace=False)
            pts = [[0.0, float(xx[i]), float(yy[i])] for i in idx]

        return np.array(pts, dtype=np.float32)

    def _filter_tracks(
        self,
        tracks: np.ndarray,
        vis: np.ndarray,
        min_vis_ratio: float = 0.3,
        max_jump_px: float = 120.0,
    ) -> tuple:
        """过滤低质量追踪轨迹。

        Args:
            tracks:        (T, N, 2)
            vis:           (T, N) bool/float，Co-Tracker 可见性
            min_vis_ratio: 至少此比例帧可见才保留该点
            max_jump_px:   单帧最大允许位移（像素），超过则认为发生跳变

        Returns:
            filtered_tracks (T, M, 2), filtered_vis (T, M)
        """
        T, N, _ = tracks.shape
        if N == 0:
            return tracks, vis

        # 1. 可见性过滤
        vis_ratio = vis.mean(axis=0)           # (N,)
        vis_ok = vis_ratio >= min_vis_ratio

        # 2. 跳变过滤：任意相邻帧位移超阈值则丢弃
        if T > 1:
            delta = np.linalg.norm(np.diff(tracks, axis=0), axis=2)   # (T-1, N)
            jump_ok = delta.max(axis=0) < max_jump_px
        else:
            jump_ok = np.ones(N, dtype=bool)

        keep = vis_ok & jump_ok
        n_removed = int((~keep).sum())
        if n_removed > 0:
            pdi_logger.info(f"追踪过滤: 丢弃 {n_removed}/{N} 个低质量点 "
                            f"(可见性不足:{int((~vis_ok).sum())} 跳变:{int((~jump_ok).sum())})")

        # 至少保留 2 个点；全部不可信时 fallback 保留全部并警告
        if keep.sum() < 2:
            pdi_logger.warning(f"追踪质量极低：所有 {N} 个点均不可信，保留全部以防崩溃")
            keep = np.ones(N, dtype=bool)

        return tracks[:, keep, :], vis[:, keep]

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