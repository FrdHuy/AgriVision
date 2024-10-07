import os
import cv2
from ultralytics import YOLO

# TODO: 单纯的检测，可能考虑删除
def detect_and_save_results(model, image_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image: {image_path}")

        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue

        # 进行推理，获取边界框
        results = model.predict(source=image)

        # 将检测结果可视化
        for result in results:
            # 绘制带有检测框的图像
            result_img = result.plot()

            # 将处理后的图像保存到输出文件夹
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, result_img)
            print(f"Saved annotated image to {output_path}")

if __name__ == "__main__":
    # 模型文件路径
    model_path = "../models/onion_detection/weights/best.pt"

    # 洋葱图片的文件夹路径
    image_folder_path = "../data/output/onion_frames/"

    # 输出文件夹路径
    output_folder_path = "../results/detections/"

    # 加载训练好的模型
    model = YOLO(model_path)

    # 获取边界框并保存处理结果
    detect_and_save_results(model, image_folder_path, output_folder_path)
