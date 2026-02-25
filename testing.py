import cv2

def extract_first_frame(video_path, output_image_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened())
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 读取第一帧
    ret, frame = cap.read()
    
    if ret:
        # 保存第一帧为图片
        cv2.imwrite(output_image_path, frame)
        print(f"First frame saved to {output_image_path}")
    else:
        print("Error: Could not read the first frame.")
    
    # 释放视频对象
    cap.release()

# 使用示例
video_path = "train_track.mp4"  # 替换为视频文件的路径
output_image_path = "first_frame_1.jpg"  # 输出保存第一帧的文件路径

extract_first_frame(video_path, output_image_path)