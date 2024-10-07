import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import math

def extract_onion_body(image, box, image_file, output_folder):
    # 提取边界框中的区域（ROI）
    x_min, y_min, x_max, y_max = map(int, box)
    roi = image[y_min:y_max, x_min:x_max]

    # 转换为灰度图像
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu 阈值
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反转图像，使洋葱本体为白色
    thresh = cv2.bitwise_not(thresh)

    # 形态学操作，去除噪声
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 找到轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 存储满足条件的轮廓
    candidate_contours = []

    for cnt in contours:
        # 计算轮廓的面积和周长
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue  # 避免除以零

        # 计算圆度
        circularity = 4 * math.pi * (area / (perimeter * perimeter))

        # 设置面积和圆度的阈值，需根据实际情况调整
        if area > 1000 and 0.5 < circularity < 1.2:
            candidate_contours.append((cnt, circularity))

    if candidate_contours:
        # 按圆度排序，选择最接近圆的轮廓
        candidate_contours.sort(key=lambda x: abs(x[1] - 1.0))
        best_contour = candidate_contours[0][0]

        # 获取洋葱本体的边界框
        x, y, w, h = cv2.boundingRect(best_contour)
        onion_body_width = w  # 洋葱本体的宽度

        # 可视化结果
        output_image_path = os.path.join(output_folder, f"processed_{image_file}")
        result_img = roi.copy()
        cv2.drawContours(result_img, [best_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(output_image_path, result_img)
        print(f"Saved processed image to {output_image_path}")

        return onion_body_width
    else:
        print(f"No suitable contour found in {image_file}")
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
