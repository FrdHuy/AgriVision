import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())   # 打印出 GPU 数量
print(torch.cuda.get_device_name(0))  # 打印出第一个 GPU 的名字
