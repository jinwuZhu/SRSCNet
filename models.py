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
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_last = nn.Conv2d(num_feat, num_ch, kernel_size=1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # -1,1,H,W
        y = self.conv_first(x)
        y = self.lrelu(y)
        y = self.res_block(y)
        y = self.lrelu(self.pixel_shuffle(self.upconv(y)))
        out = self.conv_last(self.tanh(self.conv_hr(y)))
        # out 其实学习到的是普通增强后和原图的差别
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        
        # 卷积块：Conv + BatchNorm + LeakyReLU
        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.conv_block = nn.Sequential(
            conv_block(in_channels, 64, batch_norm=False),  # (B, 1, H, W) -> (B, 64, H/2, W/2)
            conv_block(64, 128),  # (B, 64, H/2, W/2) -> (B, 128, H/4, W/4)
            conv_block(128, 256), # (B, 128, H/4, W/4) -> (B, 256, H/8, W/8)
            conv_block(256, 512), # (B, 256, H/8, W/8) -> (B, 512, H/16, W/16)
            nn.Conv2d(512, 16, kernel_size=3, stride=1, padding=1) # (B, 512, H/16, W/16) -> (B, 1, H/16, W/16)
        )
        self.liner = nn.Linear(16*16*16,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv_block(x)
        y = y.view(y.size(0),-1)
        y = self.liner(y)
        y = self.sigmoid(y)
        return y

if __name__ == '__main__':
    from torchsummary import summary
    model_D = Discriminator()
    summary(model_D,input_size=(1,256,256),device='cpu')
    exit()
    import time
    import torch.quantization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SRSCNet()
    # summary(model=model,input_size=(1,400,280),device='cpu')
    model.eval()
    input = torch.rand(size=(12,1,1280//2,720//2),device=device)
    start_time = time.time()
    with torch.no_grad():
        output = model(input)
    end_time = time.time()
    print("Time(s): ",end_time-start_time)