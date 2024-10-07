import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

def extract_onion_body(image, box, image_file, output_folder):
    # 提取边界框中的区域（ROI）
    x_min, y_min, x_max, y_max = map(int, box)
    roi = image[y_min:y_max, x_min:x_max]

    # 转换为 Lab 颜色空间
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(lab)

    # 对 L 通道进行自适应阈值处理
    thresh = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 使用形态学操作清理图像
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 取最大的轮廓（假设为洋葱本体）
        largest_contour = max(contours, key=cv2.contourArea)

        # 获取洋葱本体的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        onion_body_width = w  # 洋葱本体的宽度

        # 可视化结果
        output_image_path = os.path.join(output_folder, f"processed_{image_file}")
        result_img = cv2.drawContours(roi.copy(), [largest_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(output_image_path, result_img)
        print(f"Saved processed image to {output_image_path}")

        return onion_body_width
    return None

def measure_onion_body_width(model, image_folder, output_csv, output_folder):
    results_data = []  # 存储每张图片的处理结果

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

        # 遍历检测结果
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # 获取每个边界框并计算洋葱本体的宽度
                    onion_body_width = extract_onion_body(image, box.xyxy[0].cpu().numpy(), image_file, output_folder)
                    if onion_body_width:
                        results_data.append({"Image": image_file, "Onion Body Width": onion_body_width})

    # 将结果保存为 CSV 文件
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # 模型文件路径
    model_path = "../models/onion_detection/weights/best.pt"

    # 洋葱图片的文件夹路径
    image_folder_path = "../data/output/onion_frames/"

    # 输出处理后的图像文件夹路径
    output_processed_folder = "../results/processed_onions_width/"

    # 输出结果保存为 CSV 文件
    output_csv = "../results/onion_body_widths.csv"

    # 加载训练好的模型
    model = YOLO(model_path)

    # 处理图像并计算洋葱本体的宽度
    measure_onion_body_width(model, image_folder_path, output_csv, output_processed_folder)
