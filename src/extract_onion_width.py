import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd

def extract_onion_body(image, box, image_file, output_folder):
    # 提取边界框中的区域（ROI），包括洋葱本体和叶子
    x_min, y_min, x_max, y_max = map(int, box)
    roi = image[y_min:y_max, x_min:x_max]

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 定义白色的颜色范围 (可以根据需要调整)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # 创建遮罩，过滤出白色区域（即洋葱本体）
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 取最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 获取洋葱本体的边界框 (白色部分)
        x, y, w, h = cv2.boundingRect(largest_contour)
        onion_body_width = w  # 洋葱本体的宽度

        # 可视化洋葱本体的白色部分
        output_image_path = os.path.join(output_folder, f"processed_{image_file}")
        result_img = cv2.bitwise_and(roi, roi, mask=mask)  # 将遮罩应用到原始图像的ROI
        cv2.imwrite(output_image_path, result_img)  # 保存处理后的图像
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
