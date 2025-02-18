from torch.utils.data import  Dataset
import os
import cv2
import random

class HDImageDataset(Dataset):
    def __init__(self, image_folder:str, transform=None,crop_size:tuple[int]=(512,512),max_len:int = -1):
        self.image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('.png') or x.endswith('.jpg')]
        if(max_len >= 0 and len(self.image_paths) > max_len):
            self.image_paths=self.image_paths[:max_len]
        self.transform = transform
        self.crop_size = crop_size  # 裁剪区域的大小
    
    def __len__(self):
        return len(self.image_paths)
    
    def random_crop(self,image):
        # 确保图像尺寸大于裁剪尺寸
        assert image.shape[0] >= self.crop_size[0] and image.shape[1] >= self.crop_size[1], "图像尺寸小于裁剪尺寸"

        # 随机选择起始点坐标
        top = random.randint(0, image.shape[0] - self.crop_size[0])
        left = random.randint(0, image.shape[1] - self.crop_size[1])

        # 裁剪图像
        cropped_image = image[top: top + self.crop_size[0], left: left + self.crop_size[1]]
        return cropped_image

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        # 从IMAGE中随机裁剪一块区域
        image = self.random_crop(image)
        # 提取区域的L通道
        hls_image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
        l_channel = hls_image[:, :, 1]
        # 下采样的L通道
        l_low_channel = cv2.resize(l_channel,dsize=(image.shape[1]//2,image.shape[0]//2))
        if self.transform:
            l_channel = self.transform(l_channel)
            l_low_channel = self.transform(l_low_channel)

        return l_low_channel, l_channel  # 返回低分辨率图像和对应的高清图像

