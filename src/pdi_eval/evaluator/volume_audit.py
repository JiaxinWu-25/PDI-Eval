import numpy as np
import cv2
from typing import Optional, Tuple


def audit_rigidity_stability(
    tracks: np.ndarray,
    h_seq: np.ndarray,
    n_pairs: int = 30,
) -> Tuple[float, np.ndarray]:
    """抗旋转的刚性稳定性审计

    改进点：不再除以 h(t)（旋转时 h 会变化导致误报）。
    改用「点对距离比值协同度」：刚体缩放时，所有点对的距离应等比例缩小，
    比值方差极小；发生非物理拉伸时，各点缩放不一致，方差升高。

    定义：
        ratio_ij(t) = d_ij(t) / d_ij(0)
        score(t)    = std(ratios) / (mean(ratios) + 1e-6)  ← 比例协同失败度

    Args:
        tracks:  (T, N, 2) Co-Tracker 追踪轨迹
        h_seq:   (T,) 保留接口兼容，不再用于归一化
        n_pairs: 随机采样锚点对数量

    Returns:
        (rigidity_cv, rigidity_history)
        rigidity_cv:      float，全时段协同失败均值，越高越「果冻」
        rigidity_history: (T,) 每帧的比例协同失败度
    """
    T, N, _ = tracks.shape
    if N < 2 or T < 2:
        return 0.0, np.zeros(T)

    rng = np.random.default_rng(42)
    actual_pairs = min(n_pairs, N * (N - 1) // 2)
    pairs = []
    seen = set()
    while len(pairs) < actual_pairs:
        i, j = rng.choice(N, 2, replace=False)
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append((int(i), int(j)))

    # 第 0 帧基准距离
    first_dists = np.array([
        np.linalg.norm(tracks[0, i] - tracks[0, j]) + 1e-6
        for i, j in pairs
    ])

    rigidity_history = [1.0]  # t=0 基准帧，协同度完美
    for t in range(1, T):
        curr_dists = np.array([
            np.linalg.norm(tracks[t, i] - tracks[t, j])
            for i, j in pairs
        ])
        ratios = curr_dists / first_dists
        mean_r = float(np.mean(ratios))
        score = float(np.std(ratios)) / (mean_r + 1e-6)
        rigidity_history.append(score)

    rigidity_history = np.array(rigidity_history)
    return float(np.mean(rigidity_history)), rigidity_history


def audit_3d_volume_stability(
    pointmaps: Optional[np.ndarray],
    masks: np.ndarray,
    tracks: Optional[np.ndarray] = None,
    h_seq: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """物理体积/刚性稳定性审计（双策略）

    策略优先级：
    1. 若 pointmaps 有效（非全零）→ 3D 点云身高法
    2. 否则若 tracks + h_seq 可用   → 刚性稳定性法（推荐，不依赖 MegaSAM）
    3. 兜底                          → 返回零序列
    """
    T = len(masks)

    # --- 策略 1: 3D 点云 ---
    if pointmaps is not None:
        # 有效性检验：在第 0 帧的前景 mask 区域内判断，而非图像中心 patch
        # 条件：前景内非零点占比 > 50%（抵御世界坐标系尺度差异，不用绝对阈值）
        mask0 = masks[0]
        H_p, W_p = pointmaps.shape[1], pointmaps.shape[2]
        H_m, W_m = mask0.shape[:2]
        if (H_p, W_p) != (H_m, W_m):
            mask0_r = cv2.resize(mask0.astype(np.uint8), (W_p, H_p), interpolation=cv2.INTER_NEAREST)
        else:
            mask0_r = mask0
        fg_pts0 = pointmaps[0][mask0_r > 0]
        fg_valid = (fg_pts0.shape[0] > 0) and (np.mean(np.any(fg_pts0 != 0, axis=-1)) > 0.5)
        if fg_valid:
            vol_history = []
            for t in range(T):
                pointmap_t = pointmaps[t]
                mask_t = masks[t]
                h_p, w_p = pointmap_t.shape[:2]
                h_m, w_m = mask_t.shape[:2]
                if (h_p, w_p) != (h_m, w_m):
                    mask_t = cv2.resize(
                        mask_t.astype(np.uint8), (w_p, h_p),
                        interpolation=cv2.INTER_NEAREST
                    )
                bool_mask = mask_t > 0
                if np.any(bool_mask):
                    y_pts = pointmap_t[bool_mask][:, 1]
                    h_3d = np.percentile(y_pts, 95) - np.percentile(y_pts, 5)
                    vol_history.append(h_3d)
                else:
                    vol_history.append(vol_history[-1] if vol_history else 0.0)
            vol_history = np.array(vol_history)
            if np.mean(vol_history) > 1e-6:
                vol_cv = float(np.std(vol_history) / np.mean(vol_history))
                return vol_cv, vol_history

    # --- 策略 2: 刚性稳定性（Co-Tracker）---
    if tracks is not None and h_seq is not None:
        return audit_rigidity_stability(tracks, h_seq)

    # --- 兜底 ---
    return 0.0, np.zeros(T)
