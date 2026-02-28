"""
检测 1s_mega_sam.npz 是否包含有效的 Mega-SAM 结果（而非 fallback 占位数据）

cache 保存字段（来自 pipeline.py）：
  depth_z       (T,)        物体深度序列（归一化）
  focal_length  scalar      推断焦距
  camera_poses  (T, 4, 4)   相机 C2W 位姿矩阵
  pointmaps     (T, H, W, 3) 世界坐标系点图
"""

import numpy as np
import sys

NPZ_PATH = "output/cache/ai1_mega_sam.npz"

FALLBACK_FOCAL   = 1000.0
FALLBACK_DEPTH   = 1.0   # fallback depth_z 全是 1
FALLBACK_POINTMAP_SUM = 0.0  # fallback pointmaps 全零

def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    return ok


def main():
    print(f"Loading: {NPZ_PATH}")
    try:
        # mmap_mode='r' 延迟加载，避免将 GB 级 pointmaps 全量读入内存
        d = np.load(NPZ_PATH, allow_pickle=True, mmap_mode='r')
    except FileNotFoundError:
        print(f"[ERROR] File not found: {NPZ_PATH}")
        sys.exit(1)

    print(f"Keys: {list(d.files)}\n")

    results = []

    # --- 1. focal_length ---
    focal = float(d["focal_length"])
    print(f"focal_length = {focal:.2f}")
    results.append(check(
        "focal_length != fallback (1000.0)",
        abs(focal - FALLBACK_FOCAL) > 1.0,
        f"got {focal:.2f}"
    ))

    # --- 2. depth_z ---
    depth_z = d["depth_z"]
    print(f"\ndepth_z shape = {depth_z.shape},  range = [{depth_z.min():.4f}, {depth_z.max():.4f}]")
    depth_std = float(np.std(depth_z))
    results.append(check(
        "depth_z has variation (std > 0.01)",
        depth_std > 0.01,
        f"std={depth_std:.4f}"
    ))
    results.append(check(
        "depth_z not all-ones (fallback pattern)",
        not np.allclose(depth_z, FALLBACK_DEPTH, atol=1e-3),
        f"mean={depth_z.mean():.4f}"
    ))

    # --- 3. camera_poses ---
    poses = d["camera_poses"]
    print(f"\ncamera_poses shape = {poses.shape}")
    identity = np.eye(4)
    poses_are_identity = all(np.allclose(poses[t], identity, atol=1e-4) for t in range(len(poses)))
    results.append(check(
        "camera_poses are not all identity matrices (fallback pattern)",
        not poses_are_identity
    ))
    # 检查旋转部分是否合理（det ≈ 1）
    rot_dets = [np.linalg.det(poses[t, :3, :3]) for t in range(len(poses))]
    results.append(check(
        "rotation matrices have det ≈ 1.0 (valid SO3)",
        all(abs(det - 1.0) < 0.05 for det in rot_dets),
        f"det range [{min(rot_dets):.4f}, {max(rot_dets):.4f}]"
    ))

    # --- 4. pointmaps（只检查 shape，不读数据体——npz 是 zip 压缩，读切片需全量解压）---
    pm_shape = d["pointmaps"].shape
    T_pm, H_pm, W_pm = pm_shape[0], pm_shape[1], pm_shape[2]
    print(f"\npointmaps shape = {pm_shape},  dtype = {d['pointmaps'].dtype}")

    # fallback pointmaps 固定为 (T, H_mask, W_mask, 3)，分辨率与实际视频一致
    # 真实重建分辨率通常等于抽帧后的视频帧尺寸（不会是 0x0）
    results.append(check(
        "pointmaps shape is non-trivial (T > 0, H > 0, W > 0)",
        T_pm > 0 and H_pm > 0 and W_pm > 0,
        f"T={T_pm}, H={H_pm}, W={W_pm}"
    ))

    # --- Summary ---
    passed = sum(results)
    total  = len(results)
    print(f"\n{'='*45}")
    if passed == total:
        print(f"  RESULT: ALL {total} CHECKS PASSED — Mega-SAM produced valid 3D data")
    else:
        print(f"  RESULT: {passed}/{total} passed — Mega-SAM may have used fallback data")
        print("  Suggestion: delete output/cache/1s_mega_sam.npz and re-run main.py")
    print(f"{'='*45}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
