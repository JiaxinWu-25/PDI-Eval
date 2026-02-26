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

    def infer(self, video_path: str, initial_mask: np.ndarray, grid_size: int = 10, **kwargs) -> PerceptionResult:
        import cv2
        
        # 1. 读取视频并进行空间缩放
        cap = cv2.VideoCapture(video_path)
        frames = []
        max_dim = 880 
        
        while True:
            ret, frame = cap.read()
            if not ret: break
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
        video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None] # (1, T, 3, H, W)
        video_tensor = video_tensor.to(self.device) 

        # 缩放初始 Mask，并保证前景=1（Co-Tracker 要求非零为前景）
        small_mask = cv2.resize(initial_mask.astype(np.uint8), (curr_w, curr_h), interpolation=cv2.INTER_NEAREST)
        small_mask = (small_mask > 0).astype(np.uint8)

        # 从 mask 内均匀采样点作为 queries，避免 grid+segm_mask 导致 0 点（网格带 margin 且目标小时会筛掉所有点）
        yy, xx = np.where(small_mask > 0)
        if len(yy) == 0:
            pdi_logger.warning("初始 mask 无前景像素，改用全图网格追踪")
            queries = None
        else:
            n_pts = min(grid_size * grid_size, len(yy))
            idx = np.linspace(0, len(yy) - 1, n_pts, dtype=int)
            qx = xx[idx].astype(np.float32)
            qy = yy[idx].astype(np.float32)
            # (t, x, y)，与 CoTracker 的 queries 格式一致；t=0 表示从第 0 帧开始追
            queries_np = np.stack([np.zeros(n_pts), qx, qy], axis=1).astype(np.float32)
            queries = torch.from_numpy(queries_np[None]).to(self.device)

        pdi_logger.info(f"Co-Tracker 正在执行全视频追踪 (优化后尺寸: {curr_w}x{curr_h})...")
        
        # 3. 执行推理：有 mask 内采样点则用 queries，否则用 grid
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if queries is not None:
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        queries=queries,
                        grid_size=0,
                        grid_query_frame=0,
                    )
                else:
                    tracks, visibility = self.model(
                        video_tensor.float(),
                        grid_size=grid_size,
                        grid_query_frame=0,
                    )
        
        # 4. 坐标还原
        tracks_np = tracks[0].cpu().numpy() 
        tracks_np[:, :, 0] *= scale_x
        tracks_np[:, :, 1] *= scale_y
        
        visibility_np = visibility[0].cpu().numpy()
        breathing_metric = self.calculate_breathing_artifact(tracks_np)
        
        del video_tensor
        torch.cuda.empty_cache()
        
        return PerceptionResult(
            video_id=os.path.basename(video_path),
            frames_count=len(tracks_np),
            masks=np.zeros((1, 1, 1)), 
            h_pixel=np.zeros(len(tracks_np)),
            x_center=np.zeros(len(tracks_np)),
            tracks_2d=tracks_np,
            confidence=visibility_np,
            metadata={"breathing_metric": breathing_metric}
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