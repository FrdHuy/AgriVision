import cv2
import os

# 该文件用于分割视频文件为帧图片
def extract_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取视频文件
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    frame_interval = int(fps)  # 每秒一帧

    success, frame = video.read()
    count = 0
    frame_count = 0

    while success:
        # 只每隔 frame_interval 提取一帧
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {count} at {frame_filename}")
            count += 1

        success, frame = video.read()
        frame_count += 1

    video.release()
    print(f"Total frames extracted: {count}")

if __name__ == "__main__":
    video_file = "../data/input/onion.mov"  # 你的视频文件路径
    output_dir = "../data/output/onion_frames"  # 帧存放的输出路径
    extract_frames(video_file, output_dir)
