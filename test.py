from models import SRSCNet  # 假设SRSCNet定义在这个模块中
import torch
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
from utils import dump_frist_output

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRSCNet(num_ch=3, num_res=32, num_feat=128)
    checkpoint = torch.load('checkpoints/checkpoint_GAN_6.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    input_path = 'images/target.jpg'
    input_name = Path(input_path).stem
    output_path = f'images/{input_name}_sr.jpg'

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    original_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    # 将原图下采样取得低分辨率的测试图
    input_img = cv2.resize(original_img,dsize=(original_img.shape[1]//2, original_img.shape[0]//2))
    # 
    input_l = input_img
    input = trans(input_l).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input)
    output = output[0].squeeze().cpu().detach().numpy()  # 转换为numpy数组
    output = (output * 0.5) + 0.5  # 反归一化
    output = np.clip(output, 0, 1)  # 确保像素值在[0, 1]范围内
    sr_image = (output * 255).astype(np.uint8)  # 转换为uint8类型
    sr_image = sr_image.transpose(1, 2, 0)  # 改变形状以匹配OpenCV的期望输入

    lsr_image = cv2.resize(input_img,dsize=(input_img.shape[1]*2,input_img.shape[0]*2),interpolation=cv2.INTER_LINEAR_EXACT)
    # 保存一份普通放大的图像
    lsr_path = f'images/{input_name}_lsr.jpg'
    cv2.imwrite(lsr_path, lsr_image)
    
    # 保存超分辨率图像
    cv2.imwrite(output_path, sr_image)
    print(f"Super-resolution image saved to {output_path}")

    

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print("执行用时(s): ",end_time - start_time)