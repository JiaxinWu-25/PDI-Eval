import numpy as np
import cv2  # 需要用到 cv2 进行快速缩放
from typing import Optional, Tuple

def audit_3d_volume_stability(
    pointmaps: Optional[np.ndarray], 
    masks: np.ndarray, 
    tracks: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray]:
    """
    物理体积稳定性审计员
    已修复：解决了 Mask 与 Pointmap 分辨率不一致导致的 IndexError
    """
    T = len(masks)
    vol_history = []

    # --- 策略 1: 基于 3D 点云 ---
    if pointmaps is not None:
        pdi_logger_info = None # 内部不直接引用 logger 以保持纯净
        
        for t in range(T):
            pointmap_t = pointmaps[t] # 形状通常为 (H_p, W_p, 3)
            mask_t = masks[t]         # 形状通常为 (H_orig, W_orig)
            
            # 获取两个组件的目标分辨率
            h_p, w_p = pointmap_t.shape[:2]
            h_m, w_m = mask_t.shape[:2]

            # --- 关键修复：分辨率对齐 ---
            if (h_p, w_p) != (h_m, w_m):
                # 将大尺寸的 Mask 缩小到点云图的尺寸
                # 使用 INTER_NEAREST 确保掩码依然是 0/1 二值化，不产生模糊边缘
                mask_t_resized = cv2.resize(
                    mask_t.astype(np.uint8), 
                    (w_p, h_p), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_t_resized = mask_t

            # 转换为布尔索引
            bool_mask = mask_t_resized > 0
            
            if np.any(bool_mask):
                # 现在 bool_mask 的形状 (512, 512) 与 pointmap_t (512, 512, 3) 匹配了
                y_pts = pointmap_t[bool_mask][:, 1]
                # 过滤噪点提取高度
                h_3d = np.percentile(y_pts, 95) - np.percentile(y_pts, 5)
                vol_history.append(h_3d)
            else:
                vol_history.append(vol_history[-1] if vol_history else 0.0)

    # --- 策略 2: 基于 2D 追踪点的降级方案 ---
    elif tracks is not None:
        for t in range(T):
            pts = tracks[t]
            h_2d = np.max(pts[:, 1]) - np.min(pts[:, 1])
            vol_history.append(h_2d)

    vol_history = np.array(vol_history)
    
    if len(vol_history) == 0 or np.mean(vol_history) < 1e-6:
        return 0.0, np.zeros(T)

    # 计算变异系数 (CV)
    vol_cv = float(np.std(vol_history) / np.mean(vol_history))
    
    return vol_cv, vol_history