import os
import cv2
import pandas as pd
from ultralytics import YOLO

def calculate_onion_width(model, image_folder, output_csv, output_image_folder):
    results_data = []  # 存储结果

    # 确保输出文件夹存在
    os.makedirs(output_image_folder, exist_ok=True)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing image: {image_path}")

        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue

        # 使用模型进行推理，获取边界框
        results = model.predict(source=image)

        # 遍历检测结果，获取洋葱本体的宽度
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    # 提取边界框的左右边界，计算宽度
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    width = x_max - x_min  # 洋葱本体的宽度

                    # 存储结果
                    results_data.append({"Image": image_file, "Onion Width": width})

                    # 绘制检测框
                    result_img = result.plot()

                    # TODO: 同一张图像可能有多个洋葱，让文件只保存一次
                    # 保存带有边界框的图像到指定文件夹
                    output_image_path = os.path.join(output_image_folder, f"processed_{image_file}")
                    cv2.imwrite(output_image_path, result_img)
                    print(f"Saved annotated image to {output_image_path}")

    # 将结果保存到 CSV 文件
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # 模型文件路径
    model_path = "../models/onion_detection/weights/best.pt"

    # 洋葱图片的文件夹路径
    image_folder_path = "../data/output/onion_frames/"

    # 输出处理后的图像文件夹路径
    output_processed_folder = "../results/detections_widths/"

    # 输出结果保存为 CSV 文件
    output_csv = "../results/onion_body_widths.csv"

    # 加载训练好的模型
    model = YOLO(model_path)

    # 计算洋葱宽度并保存到 CSV 文件，同时保存检测图像
    calculate_onion_width(model, image_folder_path, output_csv, output_processed_folder)
