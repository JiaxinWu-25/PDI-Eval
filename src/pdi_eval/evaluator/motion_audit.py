import numpy as np
from typing import List, Union

def audit_trajectory_consistency(h_seq: np.ndarray, x_seq: np.ndarray, cx: float) -> float:
    """轨迹审计员：检测物体位移与缩放的空间耦合性 (H-X 齐次性)
    
    原理：ε = |(h1/hi) - (x1-cx)/(xi-cx)|
    捕捉幻觉：滑步（Skating）、地平面漂移。
    
    Args:
        h_seq: 像素高度序列 (T,)
        x_seq: 像素横坐标序列 (物体质心) (T,)
        cx: 画面中心 (Principal Point)，通常为 W/2.0
        
    Returns:
        rmse_trajectory: 轨迹残差的均方根误差 (RMSE)
    """
    if len(h_seq) < 2:
        return 0.0
        
    errors = []
    
    # 1. 提取首帧基准值 (使用前 5 帧均值增加稳定性)
    h1 = np.mean(h_seq[:5]) if len(h_seq) >= 5 else (h_seq[0] if h_seq[0] != 0 else 1.0)
    x1_rel = x_seq[0] - cx # 第一帧相对于中心点的位移
    
    # 2. 遍历序列校验
    for t in range(1, len(h_seq)):
        # 1. 尺度收敛比：物体变小了多少倍 (hi 最小设为 1.0 像素)
        hi = max(h_seq[t], 1.0)
        s_ratio = h1 / hi
        
        # 2. 横向位置收敛比：相对于消失点中心 cx 的收敛情况
        xi_rel = x_seq[t] - cx
        
        # 3. 滑步判定 (Skating Logic)
        # 如果初始位置离中心有一定距离且当前位置不在中心点
        if np.abs(x1_rel) > 5.0 and np.abs(xi_rel) > 1e-3:
            pos_ratio = x1_rel / xi_rel
            # ε_trajectory = |s_ratio - pos_ratio|
            epsilon_traj = np.abs(s_ratio - pos_ratio)
        else:
            # 物体初始位置就在中心附近，或已收敛至中心，不计入滑步残差
            epsilon_traj = 0.0
            
        errors.append(epsilon_traj)
        
    return np.array(errors) if errors else np.zeros(1)
