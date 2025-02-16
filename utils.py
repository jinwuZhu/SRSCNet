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
    cv2.imwrite(path,output_image)


###
###


def adjust_learning_rate(optimizer, shrink_factor):
    """
    调整学习率.

    :参数 optimizer: 需要调整的优化器
    :参数 shrink_factor: 调整因子，范围在 (0, 1) 之间，用于乘上原学习率.
    """

    print("\n调整学习率.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("新的学习率为 %f\n" % (optimizer.param_groups[0]['lr'], ))



class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    丢弃梯度防止计算过程中梯度爆炸.

    :参数 optimizer: 优化器，其梯度将被截断
    :参数 grad_clip: 截断值
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



import torchvision.transforms.functional as FT
def convert_image(img, source, target):
    """
    转换图像格式.

    :参数 img: 输入图像
    :参数 source: 数据源格式, 共有3种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]'
                   (3) '[-1, 1]' 
    :参数 target: 数据目标格式, 共5种
                   (1) 'pil' (PIL图像)
                   (2) '[0, 1]' 
                   (3) '[-1, 1]' 
                   (4) 'imagenet-norm' (由imagenet数据集的平均值和方差进行标准化)
                   (5) 'y-channel' (亮度通道Y，采用YCbCr颜色空间, 用于计算PSNR 和 SSIM)
    :返回: 转换后的图像
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "无法转换图像源格式 %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "无法转换图像目标格式t %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)   #把一个取值范围是[0,255]的PIL.Image 转换成形状为[C,H,W]的Tensor，取值范围是[0,1.0]

    elif source == '[0, 1]':
        pass  # 已经在[0, 1]范围内无需处理

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # 从 [0, 1] 转换至目标格式
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # 无需处理

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img

