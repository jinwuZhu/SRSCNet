import torch
import torch.nn as nn
import torch.nn.functional as F

class Shift8(nn.Module):
    def __init__(self,groups:int=4):
        
        super().__init__()
        self.g = groups
        self.shifts_list = [
            ( 1, 0), # 上
            (-1, 0), # 下
            ( 0, 1), # 左
            ( 0,-1), # 右
            ( 1, 1), # 上左
            ( 1,-1), # 上右
            (-1, 1), # 下左
            (-1,-1)  # 下右
        ]
        
    def forward(self,x):
        _,c,_,_ = x.shape

        assert c  == self.g * 8
        
        #
        x[:, 0 : 1 * self.g, :, :] = torch.roll(x[:, 0 : 1 * self.g, :, :], shifts=self.shifts_list[0], dims=(2, 3))
        x[:, 1 * self.g : 2 * self.g, :, :] = torch.roll(x[:, 1 * self.g : 2 * self.g, :, :], shifts=self.shifts_list[1], dims=(2, 3))
        x[:, 2 * self.g : 3 * self.g, :, :] = torch.roll(x[:, 2 * self.g : 3 * self.g, :, :], shifts=self.shifts_list[2], dims=(2, 3))
        x[:, 3 * self.g : 4 * self.g, :, :] = torch.roll(x[:, 3 * self.g : 4 * self.g, :, :], shifts=self.shifts_list[3], dims=(2, 3))
        x[:, 4 * self.g : 5 * self.g, :, :] = torch.roll(x[:, 4 * self.g : 5 * self.g, :, :], shifts=self.shifts_list[4], dims=(2, 3))
        x[:, 5 * self.g : 6 * self.g, :, :] = torch.roll(x[:, 5 * self.g : 6 * self.g, :, :], shifts=self.shifts_list[5], dims=(2, 3))
        x[:, 6 * self.g : 7 * self.g, :, :] = torch.roll(x[:, 6 * self.g : 7 * self.g, :, :], shifts=self.shifts_list[6], dims=(2, 3))
        x[:, 7 * self.g : 8 * self.g, :, :] = torch.roll(x[:, 7 * self.g : 8 * self.g, :, :], shifts=self.shifts_list[7], dims=(2, 3))
        return x

class ResidualBlockShift(nn.Module):
    def __init__(self,num_feat:int=64,res_scale:int = 1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat,num_feat,kernel_size=1)
        self.shift = Shift8(groups=num_feat//8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat,num_feat,kernel_size=1)
    
    def forward(self,x):
        identity = x
        out = self.conv2(self.relu(self.shift(self.conv1(x))))
        return identity + out * self.res_scale
    
class UpShiftPixelShuffle(nn.Module):
    def __init__(self, dim, scale=2) -> None:
        super().__init__()

        self.up_layer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(0.02),
            Shift8(groups=dim//8),
            nn.Conv2d(dim, dim*scale*scale, kernel_size=1),
            nn.PixelShuffle(upscale_factor=scale)
        )
    def forward(self, x):
        out = self.up_layer(x)
        return out

class UpShiftMLP(nn.Module):
    def __init__(self, dim, mode='bilinear', scale=2) -> None:
        super().__init__()

        self.up_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode=mode, align_corners=False),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.LeakyReLU(0.02),
            Shift8(groups=dim//8),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
    def forward(self, x):
        out = self.up_layer(x)
        return out

class SRSCNet(nn.Module):
    def __init__(self,num_ch:int = 1,num_res:int = 16, num_feat:int = 64):
        super().__init__()
        self.upscale = 2
        self.conv_first = nn.Conv2d(in_channels=num_ch,out_channels=num_feat,kernel_size=1)
        self.res_block = nn.Sequential(*(ResidualBlockShift(num_feat) for _ in range(num_res)))
        self.upconv = UpShiftMLP(dim=num_feat)
        self.pixel_shuffle = nn.Identity()

        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.conv_last = nn.Conv2d(num_feat, num_ch, kernel_size=1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        # -1,1,H,W
        y = self.conv_first(x)
        y = self.lrelu(y)
        y = self.res_block(y)
        y = self.lrelu(self.pixel_shuffle(self.upconv(y)))
        out = self.conv_last(self.lrelu(self.conv_hr(y)))
        # out 其实学习到的是普通增强后和原图的差别
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

if __name__ == '__main__':
    # from torchsummary import summary
    import time
    import torch.quantization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRSCNet()
    # summary(model=model,input_size=(1,400,280),device='cpu')
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,  # 要量化的模型
        dtype=torch.qint8  # 指定量化数据类型
    ).to(device)
    input = torch.rand(size=(12,1,1280//2,720//2),device=device)
    start_time = time.time()
    with torch.no_grad():
        output = quantized_model(input)
    end_time = time.time()
    print("Time(s): ",end_time-start_time)