import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import kagglehub

print("PyTorch版本:", torch.__version__)
print("Torchvision版本:", torchvision.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda if torch.cuda.is_available() else "N/A")

if torch.cuda.is_available():
    print("GPU设备:", torch.cuda.get_device_name(0))
    print("当前GPU内存使用:", torch.cuda.memory_allocated(0))