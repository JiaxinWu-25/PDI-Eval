import torch
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

from .perception.sam_wrapper import Sam2Wrapper
from .perception.track_wrapper import TrackWrapper
from .perception.mega_sam_wrapper import MegaSamWrapper
from .geometry.camera import CameraModel
from .evaluator.motion_audit import audit_trajectory_consistency
from .evaluator.scale_audit import audit_scale_consistency
from .evaluator.volume_audit import audit_3d_volume_stability
from .metrics.pdi_index import PDIIndexCalculator
from .data.cache_manager import CacheManager
from .utils.logger import pdi_logger
from .utils.visualizer import EvidenceVisualizer

class PDIEvaluationPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = CacheManager()
        self.video_id = ""
        self.video_path = ""
        self.last_report = None
        self.last_res_tracks = None

    def run(self, video_path: str, click_points: Optional[list] = None) -> Dict[str, Any]:
        self.video_path = video_path
        self.video_id = Path(video_path).stem
        pdi_logger.info(f"--- [bold blue]PDI-Eval Pipeline Start: {self.video_id}[/bold blue] ---")
        
        # 1. SAM2 Perception
        if self.cache.exists(self.video_id, "sam2"):
            pdi_logger.info("Found SAM2 cache, skipping...")
            res_2d = self.cache.load_step(self.video_id, "sam2")
        else:
            sam = Sam2Wrapper(checkpoint=self.config['sam_ckpt'], config=self.config.get('sam_cfg'))
            res_2d_raw = sam.infer(video_path, click_points=click_points)
            self.cache.save_step(self.video_id, "sam2", {
                "masks": res_2d_raw.masks,
                "h_pixel": res_2d_raw.h_pixel,
                "x_center": res_2d_raw.x_center,
                "is_truncated": res_2d_raw.is_truncated
            })
            res_2d = self.cache.load_step(self.video_id, "sam2")
            del sam
            torch.cuda.empty_cache()

        # 1.5 Co-Tracker (利用点追踪平滑轨迹并检测呼吸感)
        if self.cache.exists(self.video_id, "cotracker"):
            pdi_logger.info("Found Co-Tracker cache, skipping...")
            res_tracks = self.cache.load_step(self.video_id, "cotracker")
        else:
            tracker = TrackWrapper(checkpoint=self.config['tracker_ckpt'])
            res_tracks_raw = tracker.infer(video_path, initial_mask=res_2d['masks'][0])
            self.cache.save_step(self.video_id, "cotracker", {
                "tracks": res_tracks_raw.tracks_2d,
                "visibility": res_tracks_raw.confidence
            })
            res_tracks = self.cache.load_step(self.video_id, "cotracker")
            del tracker
            torch.cuda.empty_cache()

        # 2. 3D Engine Selection
        engine_type = self.config.get('engine_3d', 'mega_sam')
        if self.cache.exists(self.video_id, engine_type):
            pdi_logger.info(f"Found {engine_type} cache, skipping...")
            res_3d = self.cache.load_step(self.video_id, engine_type)
        else:
            perceptor = MegaSamWrapper(checkpoint=self.config.get('mega_sam_ckpt'))
            res_3d_raw = perceptor.infer(video_path, masks=res_2d['masks'])
            
            self.cache.save_step(self.video_id, engine_type, {
                "depth_z": res_3d_raw.depth_z,
                "focal_length": res_3d_raw.focal_length,
                "camera_poses": res_3d_raw.camera_poses,
                "pointmaps": res_3d_raw.pointmaps
            })
            res_3d = self.cache.load_step(self.video_id, engine_type)
            del perceptor
            torch.cuda.empty_cache()

        # 3. Geometry Alignment
        cam = CameraModel(focal_length=res_3d['focal_length'], image_size=res_2d['masks'].shape[1:])
        z_aligned = cam.align_to_unit_scale(res_3d['depth_z'])

        # 4. Evaluation Layers
        pdi_logger.info("Running Audit Layers...")
        
        # 4.1 尺度审计 (h * z)
        eps_scale_seq = audit_scale_consistency(res_2d['h_pixel'], z_aligned)
        
        # 4.2 轨迹审计 (使用 Co-Tracker 均值点以减少 Mask 抖动干扰)
        # res_tracks['tracks'] shape is (T, N, 2) -> (Frames, Points, XY)
        stable_x_seq = np.mean(res_tracks['tracks'], axis=1)[:, 0] 
        eps_traj_seq = audit_trajectory_consistency(res_2d['h_pixel'], stable_x_seq, cam.cx)
        
        # 4.3 体积审计
        vol_cv, vol_history = audit_3d_volume_stability(
            res_3d.get('pointmaps'), 
            res_2d['masks'],
            tracks=res_tracks['tracks']
        )
        
        # 5. Metrics Synthesis
        calculator = PDIIndexCalculator(
            w_scale=self.config.get('weights', {}).get('w_scale', 0.4),
            w_traj=self.config.get('weights', {}).get('w_trajectory', 0.4),
            w_vol=self.config.get('weights', {}).get('w_volume', 0.2)
        )
        final_report = calculator.compute_pdi(eps_scale_seq, eps_traj_seq, vol_cv)
        
        # 注入绘图数据
        final_report['breakdown'].update({
            'scale_history': eps_scale_seq,
            'traj_history': eps_traj_seq,
            'volume_history': vol_history
        })
        final_report['vanishing_point'] = (cam.cx, cam.cy)
        
        self.last_report = final_report
        self.last_res_tracks = res_tracks
        
        pdi_logger.pdi_report(final_report)
        return final_report

    def get_annotated_video(self):
        """生成并返回可视化视频路径。对齐视频帧数与 tracks 帧数，避免维度冲突导致未封包。"""
        if self.last_report is None or self.last_res_tracks is None:
            pdi_logger.warning("无审计结果或追踪数据，跳过标注视频生成")
            return ""
        cap = cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return ""
        frames = np.array(frames)
        tracks_T = self.last_res_tracks["tracks"]  # (T, N, 2)
        T_video, T_tracks = len(frames), tracks_T.shape[0]
        T_use = min(T_video, T_tracks)
        if T_use < T_video or T_use < T_tracks:
            pdi_logger.info(f"标注视频帧数对齐: 视频{T_video} vs 追踪{T_tracks} -> 使用{T_use}帧")
        frames = frames[:T_use]
        tracks_for_viz = tracks_T[:T_use].transpose(1, 0, 2)  # (N, T_use, 2)

        viz = EvidenceVisualizer(output_dir="results/visuals")
        annotated = viz.overlay_perspective_evidence(
            frames,
            self.last_report["vanishing_point"],
            tracks_for_viz,
            self.last_report,
        )
        out_path = viz.save_video(annotated, f"{self.video_id}_annotated.mp4")
        if out_path:
            pdi_logger.info(f"标注视频已保存: {out_path}")
        else:
            pdi_logger.warning("标注视频写入失败，请检查 results/visuals 目录与 OpenCV/FFmpeg 编码器")
        return out_path

    def get_error_plot(self):
        viz = EvidenceVisualizer(output_dir="results/visuals")
        return viz.draw_error_curves(
            self.last_report['breakdown']['scale_history'],
            self.last_report['breakdown']['traj_history'],
            self.video_id
        )