import argparse
import yaml
import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# 确保能搜到 src/ 目录下的 pdi_eval 包
sys.path.append(os.path.join(os.getcwd(), "src"))

from pdi_eval.perception.sam_wrapper import Sam2Wrapper
from pdi_eval.utils.logger import pdi_logger

def main():
    parser = argparse.ArgumentParser(description="SAM2 Only Debug Tool (Auto & Manual)")
    parser.add_argument("--input", type=str, required=True, help="输入视频路径")
    parser.add_argument("--points", type=str, default=None, help="手动点击坐标 [[x, y]]")
    parser.add_argument("--text", type=str, default=None, help="自动识别标签, 如 'train', 'car'")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="mask_check.mp4")
    args = parser.parse_args()

    # 1. 加载配置文件
    if not os.path.exists(args.config):
        pdi_logger.error(f"未找到配置文件: {args.config}")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 初始化 SAM2 Wrapper (现在包含 Florence-2 逻辑)
    pdi_logger.info("正在加载 SAM2 及其自动识别引擎...")
    sam = Sam2Wrapper(
        checkpoint=config['sam_ckpt'], 
        config=config.get('sam_cfg')
    )

    # 3. 确定目标锁定方式
    click_points = None
    if args.points:
        try:
            click_points = np.array(eval(args.points), dtype=np.float32)
            pdi_logger.info(f"使用手动点定标: {click_points}")
        except Exception as e:
            pdi_logger.error(f"坐标格式错误: {e}")
            return
    elif args.text:
        pdi_logger.info(f"使用自动文本定标: '{args.text}'")
    else:
        pdi_logger.error("必须提供 --points 或 --text 其中的一个！")
        return

    # 4. 运行推理 (内部会自动处理自动识别或手动点)
    res = sam.infer(args.input, click_points=click_points, text_query=args.text)
    
    masks = res.masks
    h_seq = res.h_pixel
    x_seq = res.x_center

    # 5. 生成可视化视频 (解决编码器问题)
    pdi_logger.info("正在渲染并合成可视化视频...")
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30 # 容错
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 关键修改：更换编码器为 XVID，这在 Linux 下重新读取的兼容性最好
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    if not out.isOpened():
        pdi_logger.error("VideoWriter 无法打开，请检查是否安装了 ffmpeg 或尝试更换 .avi 后缀")
        return

    from tqdm import tqdm
    for t in tqdm(range(len(masks)), desc="Rendering"):
        ret, frame = cap.read()
        if not ret: break

        m = masks[t].astype(np.uint8)
        if m.shape[:2] != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[m > 0] = [255, 0, 0] # 蓝色掩码
        canvas = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # 绘制中心十字
        cx = int(x_seq[t])
        cv2.drawMarker(canvas, (cx, height//2), (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
        
        # 实时信息
        cv2.putText(canvas, f"Frame: {t} | h: {int(h_seq[t])}px", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        if res.is_truncated[t]:
            cv2.putText(canvas, "TRUNCATED", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(canvas)

    cap.release()
    out.release()
    
    # 6. 文件生成后的物理检查
    if os.path.exists(args.output):
        file_size = os.path.getsize(args.output) / (1024 * 1024)
        if file_size < 0.1:
            pdi_logger.error(f"警告：生成的视频文件过小 ({file_size:.2f}MB)，可能编码失败。")
        else:
            pdi_logger.success(f"✅ 可视化完成！大小: {file_size:.2f}MB, 路径: {args.output}")
            
            # 尝试重新打开测试
            test_cap = cv2.VideoCapture(args.output)
            if test_cap.isOpened():
                pdi_logger.info("验证成功：视频可被 OpenCV 正常打开。")
                test_cap.release()
            else:
                pdi_logger.warning("验证失败：视频已生成但无法被当前 OpenCV 引擎读取，建议下载到本地查看。")
    else:
        pdi_logger.error("视频文件未能生成。")

if __name__ == "__main__":
    main()