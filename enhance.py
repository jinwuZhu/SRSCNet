from models import SRSCNet  # 假设SRSCNet定义在这个模块中
import torch
from torchvision import transforms
import cv2
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

def enhance_image(original_img,model,device):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_img = original_img
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
    return sr_image

def main():
    parser = ArgumentParser(description='图像/视频增强')
    parser.add_argument('--model','-m', type=str,default='checkpoints/checkpoint_GAN_0.pth', help='模型路径')
    parser.add_argument('--input','-i', type=str,default='', help='输入文件路径')
    parser.add_argument('--output', '-o', type=str, default='sr_image.jpg', help='输出文件路径 (默认: sr_image.jpg)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRSCNet(num_ch=3,num_res=32,num_feat=128)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    input_path:str = args.input
    output_path:str = args.output  # 超分辨率图像保存路径

    # 判断是视频还是图片
    if input_path.lower().endswith('.mp4'):
        video_reader = cv2.VideoCapture(input_path)
        input_width = video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_height = video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_reader.get(cv2.CAP_PROP_FPS), (int(input_width*2), int(input_height*2)))
        for _ in tqdm(range(total_frame_count)):
            ret, original_img = video_reader.read()
            if not ret:
                break
            sr_image = enhance_image(original_img,model,device)
            video_writer.write(sr_image)
        video_reader.release()
        video_writer.release()
    else:
        original_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        sr_image = enhance_image(original_img,model=model,device=device)
        # 保存超分辨率图像
        cv2.imwrite(output_path, sr_image)
        print(f"Super-resolution image saved to {output_path}")

    

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print("执行用时(s): ",end_time - start_time)