import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

def debug_sam2_output(npz_path, video_path=None, output_dir="debug_results"):
    print(f"正在加载缓存: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    # 1. 基础信息检查
    keys = data.files
    print(f"包含键值: {keys}")
    
    masks = data['masks']
    h_pixel = data['h_pixel']
    x_center = data['x_center']
    is_truncated = data['is_truncated']
    
    T = len(h_pixel)
    print(f"总帧数: {T}")
    print(f"平均高度: {np.mean(h_pixel):.2f} px")
    print(f"截断发生帧数: {np.sum(is_truncated)}")

    os.makedirs(output_dir, exist_ok=True)
    video_stem = Path(npz_path).stem.replace("_sam2", "")

    # 2. 绘制高度(h)和位移(x)曲线
    # 这是判断 SAM2 是否“跟丢”或“闪烁”的最快方法
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(h_pixel, label='Object Height (h)', color='blue')
    plt.title(f"SAM2 Metrics Debug: {video_stem}")
    plt.ylabel("Pixels")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x_center, label='Horizontal Center (x)', color='red')
    plt.ylabel("Pixels")
    plt.xlabel("Frame Index")
    plt.grid(True, alpha=0.3)
    plt.legend()

    curve_path = os.path.join(output_dir, f"{video_stem}_curves.png")
    plt.tight_layout()
    plt.savefig(curve_path)
    print(f"✅ 统计曲线已保存至: {curve_path}")

    # 3. 叠加可视化视频 (如果提供了原视频)
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_path = os.path.join(output_dir, f"{video_stem}_mask_check.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print("正在生成 Mask 叠加视频进行人工复核...")
        for t in range(T):
            ret, frame = cap.read()
            if not ret: break
            
            mask = masks[t].astype(np.uint8)
            # 将 Mask 放大回原图尺寸（如果缓存里存的是缩略图）
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # 创建半透明蓝色遮罩
            color_mask = np.zeros_like(frame)
            color_mask[mask > 0] = [255, 0, 0] # 蓝色
            
            # 叠加到原图
            frame = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
            
            # 绘制中心点和高度信息
            if h_pixel[t] > 0:
                cx = int(x_center[t])
                cv2.drawMarker(frame, (cx, height//2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(frame, f"h: {int(h_pixel[t])}px", (cx, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if is_truncated[t]:
                cv2.putText(frame, "TRUNCATED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            writer.write(frame)
        
        cap.release()
        writer.release()
        print(f"✅ 可视化视频已保存至: {out_path}")

if __name__ == "__main__":
    # 使用示例
    CACHE_FILE = "output/cache/demo_sam2.npz" # 替换成你的文件名
    RAW_VIDEO = "data/demo.mp4"              # 替换成你的视频路径
    
    if os.path.exists(CACHE_FILE):
        debug_sam2_output(CACHE_FILE, RAW_VIDEO)
    else:
        print(f"错误: 找不到文件 {CACHE_FILE}")