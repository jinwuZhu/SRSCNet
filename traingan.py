import torch.optim as optim
import torch
import torch.nn as nn
from models import SRSCNet, Discriminator
from datasets import HDImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import dump_frist_output

# 训练参数
image_folder = "data/Flickr2K_HR_1-2000"
num_epochs = 500
batch_size = 64
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
checkpoint_path = "" # "checkpoints/checkpoint_GAN_5.pth"
netG = SRSCNet()        # 生成器
netD = Discriminator()  # 判别器
netG.to(device)
netD.to(device)
# 加载预训练权重（如果有）
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载生成器参数
    model_state = checkpoint.get('model_state_dict')
    if model_state:
        netG.load_state_dict(model_state)
        print("已经加载生成器参数！！")

    # 加载判别器参数
    D_model_state = checkpoint.get('D_model_state_dict')
    if D_model_state:
        netD.load_state_dict(D_model_state)
        print("已加载判别器参数！！")
except FileNotFoundError:
    print("未找到预训练模型，使用随机初始化参数。")
except Exception as e:
    print(f"加载模型时发生错误: {e}")


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
])
dataset = HDImageDataset(image_folder, transform=transform, crop_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 损失函数
criterion_adv = nn.L1Loss()  # 对抗损失
criterion_l1 = nn.L1Loss()  # L1损失（保持结构）

# 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr)
optimizerG = optim.Adam(netG.parameters(), lr=lr)

# 训练循环
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader, 0):
        inputs, targets = inputs.to(device), targets.to(device)

        # --------------------
        # 训练判别器
        # --------------------

        # 将生成器切换到评估模式
        with torch.no_grad():
            fake_images = netG(inputs)

        fake_labels = torch.zeros((inputs.size(0), 1), device=device)
        real_labels = torch.ones((inputs.size(0), 1), device=device)
        netD.train()
        for _ in range(2):
            optimizerD.zero_grad()
            fake_outputs = netD(fake_images.detach())
            real_outputs = netD(targets)
            loss_D = (criterion_adv(fake_outputs,fake_labels) + criterion_adv(real_outputs,real_labels))/2
            loss_D.backward()
            optimizerD.step()

        # --------------------
        # 训练生成器
        # --------------------
        netG.train()  # 确保生成器处于训练模式
        for _ in range(4):
            optimizerG.zero_grad()
            fake_images = netG(inputs)
            output_fake_D = netD(fake_images)
            loss_fake_D = criterion_adv(output_fake_D, real_labels)  # 计算对抗损失
            # 计算L1损失（或其他重建损失）
            loss_l1 = criterion_l1(fake_images, targets)
            # 总生成器损失是L1损失加上对抗损失
            loss_G = 7 * loss_l1 + 1.2 * loss_fake_D
            # 反向传播并更新生成器参数
            loss_G.backward()
            optimizerG.step()

        print("Epoch: %d/%d; Batch: %d/%d loss_D: %.4f; loss_G: %.4f|%.4f"%(epoch + 1,num_epochs,i + 1,len(dataloader), loss_D.item(), loss_G.item(), loss_fake_D.item()))
        if(i == 0):
            dump_frist_output(fake_images,"temp/output.jpg")
            dump_frist_output(inputs,"temp/input.jpg")
            dump_frist_output(targets,"temp/target.jpg")
    # 每5个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        torch.save({'model_state_dict': netG.state_dict(), 'D_model_state_dict': netD.state_dict()}, f"checkpoints/checkpoint_GAN_{epoch+1}.pth")
        print(f"模型已保存：Epoch {epoch+1}")

print("训练完成！")