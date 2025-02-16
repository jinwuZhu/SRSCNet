from models import SRSCNet  # 假设SRSCNet定义在这个模块中
import torch
from torchvision import transforms
import cv2
import numpy as np

def postprocess_brightness(gray_img, color_img):
    # gray_img = enhance_contrast(gray_img)
    # 转换彩色图到 HLS 颜色空间
    hls_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HLS)

    # 确保灰度图大小匹配
    assert gray_img.shape == hls_img.shape[:2], "灰度图和彩色图尺寸不匹配"

    # 替换 HLS 图像的 L 通道
    hls_img[:, :, 1] = gray_img  # 直接替换 L 通道

    # 转换回 BGR 颜色空间
    adjusted_img = cv2.cvtColor(hls_img, cv2.COLOR_HLS2BGR)

    return adjusted_img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRSCNet()
    checkpoint = torch.load('checkpoints/checkpoint_epoch_89_batch_0.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    input_path = 'test.jpg'
    output_path = 'sr_image.jpg'  # 超分辨率图像保存路径

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    original_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    # 将原图下采样取得低分辨率的测试图
    input_img = cv2.resize(original_img,dsize=(original_img.shape[1]//2, original_img.shape[0]//2))
    # 取得亮度作为输入
    input_l = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    input = trans(input_l).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input)
    
    output_tensor = output.squeeze().cpu().detach().numpy()  # 转换为numpy数组
    output_tensor = (output_tensor * 0.5) + 0.5  # 反归一化
    output_tensor = np.clip(output_tensor, 0, 1)  # 确保像素值在[0, 1]范围内
    output_image = (output_tensor * 255).astype(np.uint8)  # 转换为uint8类型
    sr_image = cv2.resize(input_img,dsize=(input_img.shape[1]*2,input_img.shape[0]*2))
    # 保存一份普通放大的图像
    cv2.imwrite('lsr_image.jpg', sr_image)
    sr_image = postprocess_brightness(output_image,sr_image)
    # 保存超分辨率图像
    cv2.imwrite(output_path, sr_image)
    print(f"Super-resolution image saved to {output_path}")
    

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print("执行用时(s): ",end_time - start_time)