import numpy as np
from typing import List, Union

def audit_scale_consistency(h_seq: np.ndarray, z_seq: np.ndarray) -> float:
    """尺度审计员：检测物体缩放节奏与深度变化的物理一致性
    
    原理：h1 * z1 = h2 * z2 = constant (在针孔模型下)
    优化：增加中值滤波平滑初始状态，增强除零保护。
    
    Args:
        h_seq: 像素高度序列 (T,)
        z_seq: 3D 深度序列 (T,) (来自 Mega-SAM/Dust3R)
        
    Returns:
        rmse_scale: 尺度残差的均方根误差 (RMSE)
    """
    if len(h_seq) < 2 or len(h_seq) != len(z_seq):
        return 0.0
        
    # 1. 归一化基准处理 (使用前 5 帧中值以防止 SAM2 初始化时的噪声)
    h_ref = np.median(h_seq[:5]) if len(h_seq) >= 5 else (h_seq[0] if h_seq[0] != 0 else 1.0)
    z_ref = np.median(z_seq[:5]) if len(z_seq) >= 5 else (z_seq[0] if z_seq[0] != 0 else 1.0)
    
    h_norm = h_seq / max(h_ref, 1.0)
    z_norm = z_seq / max(z_ref, 1e-6)
    
    errors = []
    
    # 2. 遍历序列进行一致性校验 (h ∝ 1/Z)
    for t in range(1, len(h_norm)):
        # 理论缩放率：根据深度推导
        z_t = max(z_norm[t], 1e-6)
        theoretical_h_ratio = 1.0 / z_t
        
        # 实际缩放率：掩码直接观测到的高度变化
        actual_h_ratio = max(h_norm[t], 1e-6)
        
        # 3. 计算尺度残差 ε_scale
        epsilon_scale = np.abs(actual_h_ratio - theoretical_h_ratio)
        errors.append(epsilon_scale)
        
    # 返回误差序列
    return np.array(errors) if errors else np.zeros(1)
