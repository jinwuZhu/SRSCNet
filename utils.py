import cv2
import numpy as np
from torchvision import transforms

def preprocess_image(image_path, device):
    """预处理输入图像"""
    lr_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("lr_image.jpg",lr_image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    lr_image_tensor = transform(lr_image).unsqueeze(0).to(device)  # 添加batch维度，并移动到指定设备
    return lr_image_tensor, lr_image.shape  # 返回原始图像尺寸用于后续调整输出尺寸

def postprocess_output(output_tensor):
    """后处理模型输出"""
    output_tensor = output_tensor.squeeze().cpu().detach().numpy()  # 转换为numpy数组
    output_tensor = (output_tensor * 0.5) + 0.5  # 反归一化
    output_tensor = np.clip(output_tensor, 0, 1)  # 确保像素值在[0, 1]范围内
    output_image = (output_tensor * 255).astype(np.uint8)  # 转换为uint8类型

    return output_image

def dump_frist_output(output_tensor,path:str = "output.png"):
    output_tensor = output_tensor[0].squeeze().cpu().detach().numpy()  # 转换为numpy数组
    output_tensor = (output_tensor * 0.5) + 0.5  # 反归一化
    output_tensor = np.clip(output_tensor, 0, 1)  # 确保像素值在[0, 1]范围内
    output_image = (output_tensor * 255).astype(np.uint8)  # 转换为uint8类型

    # 并且颜色通道需要从RGB转换为BGR
    if len(output_image.shape) == 3 and output_image.shape[0] == 3:
        output_image = output_image.transpose(1, 2, 0)  # 改变形状以匹配OpenCV的期望输入
    
    # 保存图像
    cv2.imwrite(path, output_image)
