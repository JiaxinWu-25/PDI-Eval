import argparse
import yaml
import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# 确保能搜到 src/ 目录下的 pdi_eval 包
sys.path.append(os.path.join(os.getcwd(), "src"))

from pdi_eval.perception.sam_wrapper import Sam2Wrapper
from pdi_eval.perception.track_wrapper import TrackWrapper
from pdi_eval.utils.logger import pdi_logger

def main():
    parser = argparse.ArgumentParser(description="Co-Tracker Tracking Debug Tool (Auto & Manual)")
    parser.add_argument("--input", type=str, required=True, help="输入视频路径")
    parser.add_argument("--points", type=str, default=None, help="手动点击坐标 [[x, y]]")
    parser.add_argument("--text", type=str, default=None, help="自动识别标签, 如 'train', 'car'")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--grid_size", type=int, default=10, help="撒点密度")
    parser.add_argument("--output", type=str, default="cotracker_check.mp4")
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config):
        pdi_logger.error(f"未找到配置文件: {args.config}")
        return
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 运行 SAM2 定标 (支持自动/手动)
    pdi_logger.info("Step 1: 正在运行 SAM2 确定目标区域...")
    sam = Sam2Wrapper(checkpoint=config['sam_ckpt'], config=config.get('sam_cfg'))
    
    click_points = None
    if args.points:
        click_points = np.array(eval(args.points), dtype=np.float32)
        pdi_logger.info(f"使用手动坐标定标: {click_points}")
    elif args.text:
        pdi_logger.info(f"使用 Florence-2 自动识别: '{args.text}'")
    else:
        pdi_logger.error("错误：必须提供 --points 或 --text 其中的一个！")
        return

    # 获取包含 Mask 的结果
    res_sam = sam.infer(args.input, click_points=click_points, text_query=args.text)
    initial_mask = res_sam.masks[0] 

    # 3. 运行 Co-Tracker
    pdi_logger.info("Step 2: 正在启动 Co-Tracker 进行像素级长程追踪...")
    tracker = TrackWrapper(checkpoint=config['tracker_ckpt'])
    # 注意：这里我们调用包装好的 infer，它内部会自动处理缩放和 segm_mask
    res_tracks = tracker.infer(args.input, initial_mask=initial_mask, grid_size=args.grid_size)
    tracks = res_tracks.tracks_2d
    visibility = res_tracks.confidence

    # 4. 渲染可视化视频
    pdi_logger.info("Step 3: 正在合成追踪可视化视频...")
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    T, N, _ = tracks.shape
    # 为每个追踪点生成随机颜色
    colors = np.random.randint(0, 255, (N, 3)).tolist()

    for t in tqdm(range(T), desc="Rendering"):
        ret, frame = cap.read()
        if not ret: break

        for i in range(N):
            # 只绘制可见的点
            if visibility[t, i] < 0.5: continue

            curr_pt = (int(tracks[t, i, 0]), int(tracks[t, i, 1]))
            color = colors[i]
            
            # 绘制当前点
            cv2.circle(frame, curr_pt, 5, color, -1)
            
            # 绘制历史轨迹 (回溯 15 帧)
            if t > 0:
                for past_t in range(max(0, t-15), t):
                    p1 = (int(tracks[past_t, i, 0]), int(tracks[past_t, i, 1]))
                    p2 = (int(tracks[past_t+1, i, 0]), int(tracks[past_t+1, i, 1]))
                    cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        # UI 信息
        cv2.putText(frame, f"PDI-Eval Co-Tracker Debug | Points: {N}", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {t}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(frame)

    cap.release()
    out.release()
    pdi_logger.success(f"✅ 追踪物证已生成: {args.output}")

if __name__ == "__main__":
    main()