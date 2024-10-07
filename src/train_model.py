import os
from ultralytics import YOLO
import time

def train_yolov8(data_path, epochs, imgsz, batch_size, project_name):
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(project_root, data_path)
    model_output_path = os.path.join(project_root, 'models')

    print(f"\nProject root: {project_root}")
    print(f"Dataset path: {dataset_path}")
    print(f"Model output path: {model_output_path}\n")

    # 开始时间
    start_time = time.time()

    print("\n=== YOLOv8 Model Training ===")
    print("Loading pre-trained YOLOv8n model...")

    # 加载预训练的 YOLOv8n 模型
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully.\n")

    # 打印训练参数（通过参数动态传递）
    print("Training configuration:")
    print(f"  - Data path: {dataset_path}")
    print(f"  - Number of epochs: {epochs}")
    print(f"  - Image size: {imgsz}x{imgsz}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Project folder: {model_output_path}/{project_name}")

    # 开始训练
    print("\nTraining started...")

    try:
        model.train(
            data=dataset_path,               # 数据集的 YAML 文件路径
            epochs=epochs,                   # 训练轮次
            imgsz=imgsz,                     # 输入图像尺寸
            batch=batch_size,                # 批次大小
            name=project_name,               # 训练任务的名称
            project=model_output_path,       # 存储训练结果的路径
            verbose=True                     # 显示详细训练日志
        )
    except Exception as e:
        print(f"\nTraining failed due to an error: {e}")
        return

    # 训练结束时间
    end_time = time.time()
    training_time = end_time - start_time

    # 训练成功信息
    print("\n=== Training Completed ===")
    print(f"Total training time: {training_time:.2f} seconds.")
    print(f"Model and logs saved in: '{model_output_path}/{project_name}'\n")

if __name__ == "__main__":
    print("Starting YOLOv8 training script...")

    # 动态传入参数
    data_path = "dataset/Onion-1/data.yaml"  # 训练数据集路径
    epochs = 50                              # 训练轮次
    imgsz = 640                              # 图像大小
    batch_size = 16                          # 批次大小
    project_name = "onion_detection"         # 项目名称

    # 调用训练函数
    train_yolov8(data_path, epochs, imgsz, batch_size, project_name)
