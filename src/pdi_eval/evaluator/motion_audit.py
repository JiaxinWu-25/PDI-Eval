import numpy as np
from typing import Tuple


def audit_trajectory_consistency(
    h_seq: np.ndarray,
    xy_seq: np.ndarray,
    vanishing_point: Tuple[float, float],
) -> np.ndarray:
    """广义透视轨迹审计（横向运动自适应版）

    两种场景自动切换：
    - 纵向/斜向运动：Log(H-VP 齐次性)残差，log(h1/ht) vs log(d1/dt)
    - 横向平移（VP 在无穷远）：高度稳定性残差，|h(t) - h(0)| / h(0)

    判断依据：VP 距离序列的极差比 < 5% 视为横向运动。

    Args:
        h_seq:           (T,) SAM2 像素高度序列
        xy_seq:          (T, 2) Co-Tracker 质心坐标序列
        vanishing_point: (vx, vy)

    Returns:
        (T-1,) 轨迹残差序列
    """
    T = len(h_seq)
    if T < 2:
        return np.zeros(1)

    vp = np.array(vanishing_point, dtype=np.float64)
    dist = np.linalg.norm(xy_seq.astype(np.float64) - vp, axis=1)  # (T,)

    dist_range_ratio = float(np.ptp(dist)) / (float(np.mean(dist)) + 1e-6)

    if dist_range_ratio < 0.05:
        # 横向平移场景：深度基本不变，h 也不应变
        h0 = max(float(h_seq[0]), 1e-6)
        errors = np.abs(h_seq[1:] - h0) / h0
        return errors

    # 纵向/斜向场景：Log 空间 H-VP 齐次性
    log_h = np.log(np.maximum(h_seq, 1e-6))
    log_d = np.log(np.maximum(dist, 1e-6))

    # 用前 5 帧中值作为基准，抑制初始帧噪声
    n_ref = min(5, T)
    h_base = float(np.median(log_h[:n_ref]))
    d_base = float(np.median(log_d[:n_ref]))

    log_h_ratio = log_h - h_base
    log_d_ratio = log_d - d_base

    errors = np.abs(log_h_ratio - log_d_ratio)
    return errors[1:]
