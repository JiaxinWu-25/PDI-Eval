"""重建质量双层审计模块

第一层（数学）：低成本、极速、高严谨性
  - 地面平整度 RANSAC
  - 尺度跳变检测（Z 二阶导）
  - 重投影残差检查

第二层（MLLM 语义）：Open3D 多视角渲染 + 视觉大模型 API 裁判
  - 自动渲染俯视 / 侧视 / 45° 三张图并拼接
  - 调用 OpenAI 兼容接口（豆包 / GPT-4o 等），结构化 JSON 返回
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ..utils.logger import pdi_logger
except ImportError:
    pdi_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ------------------------------------------------------------------ #
#  第一层：数学自检                                                   #
# ------------------------------------------------------------------ #

def _ransac_plane(pts: np.ndarray, thresh: float = 0.05, iters: int = 200) -> float:
    """对 (N, 3) 点集做 RANSAC 平面拟合，返回内点 RMSE（世界单位）。"""
    N = len(pts)
    if N < 3:
        return 0.0

    rng = np.random.default_rng(42)
    best_rmse, best_n = 1e9, 0

    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        dists = np.abs(pts @ normal + d)
        inliers = dists < thresh
        n_in = int(inliers.sum())
        if n_in > best_n:
            best_n = n_in
            in_pts = pts[inliers]
            # 最小二乘精化
            A = np.column_stack([in_pts[:, :2], np.ones(len(in_pts))])
            b = in_pts[:, 2]
            try:
                coeffs, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
                pred = A @ coeffs
                best_rmse = float(np.sqrt(np.mean((b - pred) ** 2)))
            except np.linalg.LinAlgError:
                best_rmse = float(np.std(dists[inliers]))

    return best_rmse


def audit_ground_flatness(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    bottom_ratio: float = 0.2,
    rmse_threshold: float = 0.08,
) -> Tuple[float, bool]:
    """地面平整度审计：对场景底部点云做 RANSAC 平面拟合。

    Args:
        pointmaps:      (T, H, W, 3) 世界坐标点云序列
        masks:          (T, H, W) 前景 mask（排除前景区域）
        bottom_ratio:   取图像底部此比例的区域作为地面候选点
        rmse_threshold: RMSE 超过此值判定为地面不平（Scale Drift）

    Returns:
        (rmse, passed): RMSE 值 与 是否通过
    """
    T, H, W, _ = pointmaps.shape
    all_pts = []
    n_sample = min(T, 5)  # 只取前几帧，避免过慢
    y_cut = int(H * (1 - bottom_ratio))

    for t in range(n_sample):
        mask_t = masks[t] if t < len(masks) else np.zeros((H, W), dtype=np.uint8)
        if mask_t.shape != (H, W):
            mask_t = cv2.resize(mask_t.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        # 只取底部且不属于前景的区域
        region = np.zeros((H, W), dtype=bool)
        region[y_cut:, :] = True
        region &= (mask_t == 0)
        pts = pointmaps[t][region]
        valid = np.any(pts != 0, axis=-1)
        if valid.sum() > 10:
            all_pts.append(pts[valid])

    if not all_pts:
        return 0.0, True

    pts_all = np.concatenate(all_pts, axis=0)
    # 降采样避免过慢
    if len(pts_all) > 2000:
        idx = np.random.default_rng(42).choice(len(pts_all), 2000, replace=False)
        pts_all = pts_all[idx]

    rmse = _ransac_plane(pts_all)
    passed = rmse < rmse_threshold
    pdi_logger.info(f"地面平整度 RANSAC RMSE={rmse:.4f} ({'pass' if passed else 'FAIL'})")
    return rmse, passed


def audit_scale_jump(
    depth_z: np.ndarray,
    masks: np.ndarray,
    jump_threshold: float = 0.5,
) -> Tuple[float, bool]:
    """尺度跳变检测：计算前景区域中值深度的二阶差分，检测瞬时偏移。

    Args:
        depth_z:        (T, H, W) 深度图序列
        masks:          (T, H, W) 前景 mask
        jump_threshold: 二阶差分最大绝对值超过此值判定为跳变

    Returns:
        (max_jump, passed)
    """
    T = depth_z.shape[0]
    if T < 3:
        return 0.0, True

    z_median = []
    for t in range(T):
        m = masks[t] if t < len(masks) else np.zeros(depth_z.shape[1:], dtype=np.uint8)
        if m.shape != depth_z[t].shape:
            m = cv2.resize(m.astype(np.uint8), (depth_z.shape[2], depth_z.shape[1]),
                           interpolation=cv2.INTER_NEAREST)
        fg_z = depth_z[t][m > 0]
        z_median.append(float(np.median(fg_z)) if len(fg_z) > 0 else 0.0)

    z_arr = np.array(z_median)
    # 归一化到 [0, 1] 再求二阶差分，消除绝对尺度影响
    z_range = float(np.ptp(z_arr)) + 1e-6
    z_norm = (z_arr - z_arr.min()) / z_range
    d2 = np.abs(np.diff(z_norm, n=2))
    max_jump = float(d2.max()) if len(d2) > 0 else 0.0
    passed = max_jump < jump_threshold
    pdi_logger.info(f"尺度跳变检测 max_d2Z={max_jump:.4f} ({'pass' if passed else 'FAIL'})")
    return max_jump, passed


def audit_reprojection_residual(
    residuals: Optional[np.ndarray],
    threshold: float = 2.0,
) -> Tuple[float, bool]:
    """重投影残差检查：读取 MegaSAM 输出的残差序列均值。

    Args:
        residuals:  (T,) 每帧重投影残差（像素），若为 None 则跳过
        threshold:  均值超过此像素数判定为追踪失败

    Returns:
        (mean_residual, passed)
    """
    if residuals is None or len(residuals) == 0:
        return 0.0, True
    mean_res = float(np.mean(residuals))
    passed = mean_res < threshold
    pdi_logger.info(f"重投影残差均值={mean_res:.4f}px ({'pass' if passed else 'FAIL'})")
    return mean_res, passed


def audit_reconstruction_math(
    pointmaps: Optional[np.ndarray],
    depth_z: np.ndarray,
    masks: np.ndarray,
    residuals: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """第一层数学审计，汇总三项指标。

    Returns:
        dict 包含:
          ground_rmse, ground_pass,
          scale_jump, scale_jump_pass,
          reprojection_residual, reprojection_pass,
          math_pass (三项全部通过才为 True)
    """
    result: Dict[str, Any] = {}

    # 地面平整度
    if pointmaps is not None and pointmaps.ndim == 4:
        g_rmse, g_pass = audit_ground_flatness(pointmaps, masks)
    else:
        g_rmse, g_pass = 0.0, True
    result["ground_rmse"]  = round(g_rmse, 4)
    result["ground_pass"]  = g_pass

    # 尺度跳变
    sj, sj_pass = audit_scale_jump(depth_z, masks)
    result["scale_jump"]      = round(sj, 4)
    result["scale_jump_pass"] = sj_pass

    # 重投影残差
    rr, rr_pass = audit_reprojection_residual(residuals)
    result["reprojection_residual"] = round(rr, 4)
    result["reprojection_pass"]     = rr_pass

    result["math_pass"] = g_pass and sj_pass and rr_pass
    return result


# ------------------------------------------------------------------ #
#  第二层：Open3D 渲染 + MLLM 语义审计                               #
# ------------------------------------------------------------------ #

_MLLM_PROMPT = (
    "You are a Senior 3D Computer Vision Expert specializing in SfM (Structure-from-Motion) and sparse point clouds. "
    "The image shows THREE views of a 3D reconstruction from a monocular video.\n\n"
    "### CONTEXT FOR EVALUATION ###\n"
    "The reconstruction is produced by a monocular depth-aware system (Mega-SAM). "
    "EXPECTED ARTIFACTS: Gaps, missing textures, and sparse points are NORMAL and should NOT be penalized. "
    "CRITICAL FAILURES: Dimensional collapse (cardboard effect) and geometric warping (bowl effect) are FATAL errors.\n\n"
    "### VIEW DESCRIPTION ###\n"
    "- LEFT: 45-degree perspective特写. Focus on whether the object looks like a 3D volume.\n"
    "- MIDDLE: Side view特写. Focus on thickness; a successful object must NOT look paper-thin.\n"
    "- RIGHT: Top-down Bird's Eye View. Focus on the ground/track; it must be a straight plane, not curved.\n\n"
    "### EVALUATION CRITERIA ###\n"
    "1. [Object Integrity]: Does the main object maintain a consistent 3D shape across views? Even with holes, is it a recognizable 'solid' subject?\n"
    "2. [Volume vs. Paper]: In the SIDE view, does the object have depth? (Success = thick volume; Failure = flat cardboard).\n"
    "3. [Geometric Stability]: Is the ground reconstructed as a flat horizontal plane? Reject if you see 'The Bowl Effect' (ground curving like a spoon).\n\n"
    "Reply STRICTLY in the following JSON format:\n"
    '{"reconstruction_success": <bool>, "reason": "<one concise sentence focusing on geometry, ignoring texture holes>", "score": <1-10>}'
)

def _build_colored_pointcloud(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    max_pts: int = 100000,
    sample_frames: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """聚合全量点云坐标与颜色（Global Map）。均匀采样 sample_frames 帧叠加。"""
    T, H, W, _ = pointmaps.shape
    # 均匀采样覆盖全部时间范围
    frame_indices = np.linspace(0, T - 1, min(T, sample_frames), dtype=int).tolist()
    pts_list, rgb_list = [], []
    for t in frame_indices:
        m = masks[t] if t < len(masks) else np.zeros((H, W), dtype=np.uint8)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        valid = np.any(pointmaps[t] != 0, axis=-1)
        pts = pointmaps[t][valid]
        if not len(pts):
            continue
        pts_list.append(pts)
        if frames and t < len(frames) and frames[t] is not None:
            frame = frames[t]
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb_list.append(rgb[valid])
        else:
            # 按深度 Z 着色（蓝→绿→黄）
            z = pts[:, 2]
            t_val = (z - z.min()) / (z.ptp() + 1e-6)
            col = np.column_stack([t_val, 0.5 + t_val * 0.5, 1 - t_val])
            rgb_list.append(np.clip(col, 0, 1).astype(np.float32))

    if not pts_list:
        return np.empty((0, 3)), np.empty((0, 3))

    all_pts = np.concatenate(pts_list, axis=0)
    all_rgb = np.concatenate(rgb_list, axis=0)
    if len(all_pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(all_pts), max_pts, replace=False)
        all_pts, all_rgb = all_pts[idx], all_rgb[idx]
    return all_pts, all_rgb


def _render_views_matplotlib(
    all_pts: np.ndarray,
    all_rgb: np.ndarray,
    img_size: int = 512,
) -> Optional[np.ndarray]:
    """用 matplotlib 渲染三视角散点图，headless 可靠。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # MegaSAM 坐标：X 右，Y 下，Z 前；转为可视化坐标：X 右，Y 前，Z 上
    px_plot, py_plot, pz_plot = all_pts[:, 0], all_pts[:, 2], -all_pts[:, 1]
    # 等比轴范围，避免拉伸变形
    half = max(px_plot.ptp(), py_plot.ptp(), pz_plot.ptp()) / 2 + 1e-6
    cx, cy, cz = px_plot.mean(), py_plot.mean(), pz_plot.mean()

    view_params = [
        (90, -90, "Top"),
        (0,    0, "Side"),
        (30,  45, "45deg"),
    ]
    px = img_size / 100
    fig = plt.figure(figsize=(px * 3, px), dpi=100)
    for i, (elev, azim, title) in enumerate(view_params):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(px_plot, py_plot, pz_plot, c=all_rgb, s=0.3, alpha=0.7, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (img_size * 3, img_size))


def _render_views_open3d(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    img_size: int = 512,
) -> Optional[np.ndarray]:
    """生成白底三视角点云拼图（img_size*3 × img_size），发给 MLLM 审计。

    LEFT  (view1): 中间单帧正面 45°，主体特写
    MIDDLE(view2): 中间段 5 帧侧视，判断厚度/深度
    RIGHT (view3): 全局稀疏俯视，判断地面平整度与轨迹
    """
    import sys
    T, H, W, _ = pointmaps.shape

    def _build_pts(t_indices, sparse_ratio: float = 1.0):
        pts_l, col_l = [], []
        for t in t_indices:
            valid = np.any(pointmaps[t] != 0, axis=-1)
            if not valid.any():
                continue
            p = pointmaps[t][valid]
            if frames and t < len(frames) and frames[t] is not None:
                fr = frames[t]
                if fr.shape[:2] != (H, W):
                    fr = cv2.resize(fr, (W, H))
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                c = rgb[valid]
            else:
                z = p[:, 2]
                t_val = (z - z.min()) / (z.ptp() + 1e-6)
                c = np.clip(np.column_stack(
                    [t_val, 0.5 + t_val * 0.5, 1.0 - t_val]), 0, 1).astype(np.float32)
            pts_l.append(p)
            col_l.append(c)
        if not pts_l:
            return None, None
        all_p = np.concatenate(pts_l)
        all_c = np.concatenate(col_l)
        if sparse_ratio < 1.0:
            n = max(300, int(len(all_p) * sparse_ratio))
            if len(all_p) > n:
                idx = np.random.default_rng(42).choice(len(all_p), n, replace=False)
                all_p, all_c = all_p[idx], all_c[idx]
        return all_p, all_c

    mid = T // 2
    win = min(3, T // 4)
    single_idx = [min(mid, T - 1)]
    window_idx = np.linspace(max(0, mid - win), min(T - 1, mid + win), 5, dtype=int).tolist()
    global_idx  = np.linspace(0, T - 1, min(T, 30), dtype=int).tolist()

    pts_single, col_single = _build_pts(single_idx)
    pts_window, col_window = _build_pts(window_idx)
    pts_global, col_global = _build_pts(global_idx, sparse_ratio=0.05)

    if pts_single is None and pts_global is None:
        return None

    # 三视角配置（MegaSAM 坐标: X=右, Y=下, Z=前）
    view_specs = [
        dict(pts=pts_single, col=col_single, front=[0,   -0.2, -1  ], up=[0, -1,  0], zoom=0.5),
        dict(pts=pts_window, col=col_window, front=[1,   -0.2, -0.3], up=[0, -1,  0], zoom=0.5),
        dict(pts=pts_global, col=col_global, front=[0,   -1,    0.1], up=[0,  0, -1], zoom=0.8),
    ]

    try:
        import open3d as o3d
        is_win = sys.platform == "win32"
        rendered = []

        for spec in view_specs:
            p, c = spec["pts"], spec["col"]
            if p is None:
                rendered.append(np.full((img_size, img_size, 3), 255, dtype=np.uint8))
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(p)
            pcd.colors = o3d.utility.Vector3dVector(c)
            center = np.asarray(pcd.get_center())

            if not is_win:
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                mat.point_size = 8.0
                extent = float(np.linalg.norm(
                    np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())))
                renderer = o3d.visualization.rendering.OffscreenRenderer(img_size, img_size)
                renderer.scene.add_geometry("pcd", pcd, mat)
                renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
                front = np.array(spec["front"], dtype=float)
                eye = (front / np.linalg.norm(front)) * extent * spec["zoom"] * 3 + center
                renderer.scene.camera.look_at(center.tolist(), eye.tolist(), spec["up"])
                rendered.append(np.asarray(renderer.render_to_image())[:, :, :3])
            else:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False, width=img_size, height=img_size)
                vis.add_geometry(pcd)
                opt = vis.get_render_option()
                opt.point_size = 8.0
                opt.background_color = np.array([1.0, 1.0, 1.0])
                vis.poll_events()
                vis.update_renderer()
                vis.reset_view_point(True)
                ctr = vis.get_view_control()
                ctr.set_lookat(center.tolist())
                ctr.set_front(spec["front"])
                ctr.set_up(spec["up"])
                ctr.set_zoom(spec["zoom"])
                vis.poll_events()
                vis.update_renderer()
                img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
                vis.destroy_window()
                rendered.append((img * 255).astype(np.uint8))

        pdi_logger.info("open3d 白底三视角渲染成功")
        return np.concatenate(rendered, axis=1)

    except Exception as e:
        pdi_logger.warning(f"open3d 渲染失败，降级到 matplotlib: {e}")

    fb_pts, fb_rgb = _build_colored_pointcloud(pointmaps, masks, frames)
    return _render_views_matplotlib(fb_pts, fb_rgb, img_size)


def _load_video_frames_ffmpeg(
    video_path: str,
    max_frames: int = 60,
) -> List[np.ndarray]:
    """用 ffmpeg pipe 读取视频帧（BGR），绕过 OpenCV/GStreamer 编解码限制。"""
    import subprocess
    video_path = str(Path(video_path).resolve())
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pdi_logger.warning(f"ffprobe 无法读取视频: {video_path}")
        return []
    parts = probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])
    total = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else None

    # 均匀采样：先全部读入再抽帧（视频短时直接全读）
    cmd = ["ffmpeg", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    all_frames: List[np.ndarray] = []
    frame_size = h * w * 3
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        all_frames.append(np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy())
    proc.stdout.close()
    proc.wait()

    if not all_frames:
        return []
    if len(all_frames) <= max_frames:
        return all_frames
    idx = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
    return [all_frames[i] for i in idx]


def _encode_image_b64(img: np.ndarray) -> str:
    """将 RGB ndarray 编码为 base64 JPEG 字符串。"""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def audit_reconstruction_mllm(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    api_key: str,
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    model: str = "doubao-vision-pro-32k-250615",
    save_render_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """第二层 MLLM 语义审计。

    Args:
        pointmaps:        (T, H, W, 3) 点云序列
        masks:            (T, H, W) 前景 mask
        api_key:          API 密钥（豆包/OpenAI 等 OpenAI 兼容接口）
        base_url:         API endpoint
        model:            视觉模型名称
        save_render_path: 若非 None，将拼接渲染图保存到此路径（便于 debug）

    Returns:
        dict 包含:
          render_success, reconstruction_success, reason, score, mllm_raw
    """
    result: Dict[str, Any] = {
        "render_success": False,
        "reconstruction_success": None,
        "reason": "",
        "score": None,
        "mllm_raw": "",
    }

    # 渲染
    composite = _render_views_open3d(pointmaps, masks, frames)
    if composite is None:
        pdi_logger.warning("MLLM 审计跳过：渲染失败或 open3d 不可用")
        return result
    result["render_success"] = True

    if save_render_path:
        bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_render_path, bgr)
        pdi_logger.info(f"渲染图已保存: {save_render_path}")

    # # 调用 API
    try:
        from openai import OpenAI
    except ImportError:
        pdi_logger.warning("openai 包未安装，跳过 MLLM API 调用")
        return result

    b64 = _encode_image_b64(composite)
    client = OpenAI(api_key=api_key, base_url=base_url)

    pdi_logger.info(f"正在调用 MLLM 语义审计 ({model})...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _MLLM_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=256,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        result["mllm_raw"] = raw

        # 解析 JSON
        parsed = json.loads(raw)
        result["reconstruction_success"] = bool(parsed.get("reconstruction_success", True))
        result["reason"] = str(parsed.get("reason", ""))
        result["score"]  = int(parsed.get("score", 5))
        pdi_logger.info(
            f"MLLM 裁判: success={result['reconstruction_success']} "
            f"score={result['score']} reason={result['reason']}"
        )
    except json.JSONDecodeError:
        pdi_logger.warning(f"MLLM 返回非 JSON 格式: {raw[:100]}")
    except Exception as e:
        pdi_logger.warning(f"MLLM API 调用失败: {e}")

    return result


# ------------------------------------------------------------------ #
#  统一入口                                                           #
# ------------------------------------------------------------------ #

def audit_reconstruction(
    pointmaps: Optional[np.ndarray],
    depth_z: np.ndarray,
    masks: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    mllm_config: Optional[Dict[str, Any]] = None,
    save_render_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """双层重建质量审计统一入口。

    Args:
        pointmaps:        (T, H, W, 3) MegaSAM 点云序列，可为 None
        depth_z:          (T, H, W) 深度图序列
        masks:            (T, H, W) 前景 mask
        residuals:        (T,) MegaSAM 重投影残差，可为 None
        mllm_config:      dict，包含 api_key / base_url / model 等字段；
                          为 None 则跳过第二层
        save_render_path: 调试用，将渲染图保存到该路径

    Returns:
        {
          "math": {ground_rmse, scale_jump, reprojection_residual, math_pass, ...},
          "mllm": {render_success, reconstruction_success, reason, score, ...} | None,
          "overall_pass": bool,
        }
    """
    math_result = audit_reconstruction_math(pointmaps, depth_z, masks, residuals)

    mllm_result = None
    if mllm_config and mllm_config.get("api_key") and pointmaps is not None:
        if math_result["math_pass"]:
            # 数学通过后再调 MLLM，节省成本
            mllm_result = audit_reconstruction_mllm(
                pointmaps=pointmaps,
                masks=masks,
                api_key=mllm_config["api_key"],
                base_url=mllm_config.get("base_url", "https://ark.cn-beijing.volces.com/api/v3"),
                model=mllm_config.get("model", "doubao-vision-pro-32k-250615"),
                save_render_path=save_render_path,
                frames=frames,
            )
        else:
            pdi_logger.info("数学层审计未通过，跳过 MLLM 调用以节省费用")

    mllm_pass = (
        mllm_result is None                               # 未配置 → 不参与判决
        or mllm_result["reconstruction_success"] is None  # API 失败 → 弃权
        or mllm_result["reconstruction_success"]
    )

    return {
        "math": math_result,
        "mllm": mllm_result,
        "overall_pass": math_result["math_pass"] and mllm_pass,
    }


# ------------------------------------------------------------------ #
#  从 .npz + 视频加载数据                                             #
# ------------------------------------------------------------------ #

def load_from_npz(
    npz_path: str,
    video_path: Optional[str] = None,
    max_frames: int = 60,
) -> Dict[str, Any]:
    """从 MegaSAM .npz 文件（以及可选视频）加载审计所需数据。

    Args:
        npz_path:   MegaSAM 输出的 .npz 路径，应包含 pointmaps / camera_poses
        video_path: 对应原始视频路径，用于提取 RGB 帧（可选）
        max_frames: 最多使用的帧数，超出则均匀降采样

    Returns:
        dict 包含: pointmaps, depth_z, masks, residuals, frames(BGR)
    """
    data = np.load(npz_path, allow_pickle=True)
    print(f"npz keys: {list(data.keys())}")
    for k in data.keys():
        arr = data[k]
        print(f"  {k}: shape={getattr(arr, 'shape', None)}")

    # ---------- pointmaps ----------
    pointmaps = None
    if "pointmaps" in data:
        pointmaps = np.asarray(data["pointmaps"])  # (T, H, W, 3)

    # ---------- depth_z ----------
    if "depth_z" in data:
        depth_z = np.asarray(data["depth_z"])
        # 若 depth_z 是标量序列 (T,) 而非深度图 (T,H,W)，则改用 pointmaps Z 分量
        if depth_z.ndim < 3 and pointmaps is not None:
            print(f"[info] depth_z shape={depth_z.shape}，改用 pointmaps[:,:,:,2] 作为深度图")
            depth_z = pointmaps[:, :, :, 2]
    elif pointmaps is not None:
        depth_z = pointmaps[:, :, :, 2]
    else:
        raise ValueError("npz 中既无 depth_z 也无 pointmaps，无法运行审计")

    # ---------- masks ----------
    if "masks" in data:
        masks = np.asarray(data["masks"]).astype(np.uint8)
    elif "mask" in data:
        masks = np.asarray(data["mask"]).astype(np.uint8)
    else:
        T, H, W = depth_z.shape[:3]
        masks = np.zeros((T, H, W), dtype=np.uint8)
        print("[警告] npz 中无 masks 字段，使用全零占位（无前景区域）")

    # ---------- residuals ----------
    residuals = None
    for k in ("residuals", "reprojection_residuals", "reproj_err"):
        if k in data:
            residuals = np.asarray(data[k])
            break

    # ---------- 降采样帧数 ----------
    T = depth_z.shape[0]
    if T > max_frames:
        idx = np.linspace(0, T - 1, max_frames, dtype=int)
        depth_z = depth_z[idx]
        masks   = masks[idx]
        if pointmaps is not None:
            pointmaps = pointmaps[idx]
        if residuals is not None:
            residuals = residuals[idx]

    # ---------- 视频 RGB 帧（ffmpeg pipe，绕过 GStreamer 限制）----------
    frames = []
    if video_path and Path(video_path).exists():
        frames = _load_video_frames_ffmpeg(video_path, max_frames=max_frames)
        print(f"从视频读取 {len(frames)} 帧")

    return dict(
        pointmaps=pointmaps,
        depth_z=depth_z,
        masks=masks,
        residuals=residuals,
        frames=frames,
    )


# ------------------------------------------------------------------ #
#  白色背景三视角渲染                                                 #
# ------------------------------------------------------------------ #

def render_three_views_white_bg(
    pointmaps: np.ndarray,
    masks: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    output_dir: Optional[str] = None,
    img_w: int = 800,
    img_h: int = 600,
    point_size: float = 6.0,
    sample_frames: int = 20,
) -> List[str]:
    """渲染白色背景的三视角静态图（风格类似示例：方块感、白底）。

    三视角（MegaSAM 坐标 X=右 Y=下 Z=前）：
      view1 — 俯视+前倾（鸟瞰带透视感）
      view2 — 右侧视（略仰角）
      view3 — 斜后下方对角

    Returns:
        已保存的图片路径列表 [view1.png, view2.png, view3.png]
    """
    import sys

    all_pts, all_col = _build_colored_pointcloud(
        pointmaps, masks, frames,
        max_pts=300000,
        sample_frames=sample_frames,
    )
    if len(all_pts) == 0:
        pdi_logger.warning("render_three_views_white_bg: 无有效点，跳过渲染")
        return []

    try:
        import open3d as o3d
    except ImportError:
        pdi_logger.warning("open3d 未安装，无法渲染白底三视角")
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_col)
    center = np.asarray(pcd.get_center())

    view_specs = [
        dict(name="view1", front=[0.1, -0.9,  0.2],  up=[0,  0, -1], zoom=0.4),
        dict(name="view2", front=[1.0, -0.2, -0.2],  up=[0, -1,  0], zoom=0.35),
        dict(name="view3", front=[0.6,  0.25, 0.75], up=[0, -1,  0], zoom=0.35),
    ]

    if output_dir is None:
        output_dir = "."
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    saved = []
    for spec in view_specs:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=img_w, height=img_h)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([1.0, 1.0, 1.0])
        vis.poll_events()
        vis.update_renderer()
        vis.reset_view_point(True)
        ctr = vis.get_view_control()
        ctr.set_lookat(center.tolist())
        ctr.set_front(spec["front"])
        ctr.set_up(spec["up"])
        ctr.set_zoom(spec["zoom"])
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        vis.destroy_window()

        img_u8 = (img * 255).astype(np.uint8)
        out_path = str(Path(output_dir) / f"{spec['name']}.png")
        import cv2 as _cv2
        _cv2.imwrite(out_path, _cv2.cvtColor(img_u8, _cv2.COLOR_RGB2BGR))
        saved.append(out_path)
        pdi_logger.info(f"白底视角已保存: {out_path}")

    return saved


# ------------------------------------------------------------------ #
#  独立测试入口                                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    NPZ_PATH   = "scale_perspective/video/DAVIS/480p/bear_mega_sam.npz"
    VIDEO_PATH = "scale_perspective/video/DAVIS/480p/bear.mp4"

    # MLLM 配置（留空则只运行数学层）
    # MLLM_CONFIG: Optional[Dict[str, Any]] = None
    MLLM_CONFIG = {
        "api_key": "10a00d5a-a37c-4a7e-8fb3-7d8e3a3e6d66",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-2-0-pro-260215",
    }

    # 结果输出目录：results/{video_stem}/
    video_stem = Path(VIDEO_PATH).stem  # e.g. "7_bottle"
    out_dir = Path("scale_perspective/results") / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    render_path = str(out_dir / "render.jpg") if MLLM_CONFIG else None

    print(f"Loading: {NPZ_PATH}")
    loaded = load_from_npz(NPZ_PATH, VIDEO_PATH)

    result = audit_reconstruction(
        pointmaps=loaded["pointmaps"],
        depth_z=loaded["depth_z"],
        masks=loaded["masks"],
        residuals=loaded["residuals"],
        mllm_config=MLLM_CONFIG,
        save_render_path=render_path,
        frames=loaded["frames"],
    )

    # 保存完整审计 JSON
    json_path = out_dir / "audit_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"审计结果已保存: {json_path}")

    # 单独保存 mllm_raw 原始返回（便于 debug）
    if result.get("mllm") and result["mllm"].get("mllm_raw"):
        raw_path = out_dir / "mllm_raw.txt"
        raw_path.write_text(result["mllm"]["mllm_raw"], encoding="utf-8")
        print(f"MLLM 原始返回已保存: {raw_path}")

    print("\n===== Audit Result =====")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
