import math
import time
import torch
import os
import pandas as pd
from tqdm import tqdm
from math import log10
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from loss import GeneratorLoss
from models import Discriminator, SRSCNet
from torchvision import transforms
from datasets import HDImageDatasetN3
from ssim import SSIM
from utils import dump_frist_output

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='图像增强')
    parser.add_argument('--checkpoint','-c', type=str,default='checkpoints/checkpoint_GAN_19.pth', help='从指定的检查点文件继续')
    parser.add_argument('--datafolder', '-d', type=str, default= 'data/archive/test/images', help='训练数据集路径')
    parser.add_argument('--testfolder', '-t', type=str, default= 'data/archive/test/images', help='测试数据集路径')
    parser.add_argument('--epochs', '-e', type=int, default=30, help='训练次数')
    parser.add_argument('--batch', '-b', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch
    train_dataset_folder = args.datafolder
    test_dataset_folder = args.testfolder
    checkpoint_path = args.checkpoint
    checkpoints_save_folder = "checkpoints"

    # 创建用于保存训练结果的目录
    if not os.path.exists(checkpoints_save_folder):
        os.makedirs(checkpoints_save_folder)
    
    #gpu还是cpu
    device = args.device
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    # else:
    #     device = "cpu"

    input_size = 256

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道归一化
    ])
    #构建数据集
    train_dataset = HDImageDatasetN3(train_dataset_folder,transform=transform, crop_size=(input_size, input_size),max_len=400)
    #加快训练设置了<num_workers，pin_memory，drop_last>资源不足可以都删除掉
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    valid_dataset = HDImageDatasetN3(test_dataset_folder,transform=transform, crop_size=(input_size, input_size),max_len=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    # 创建生成器模型对象Generator
    netG = SRSCNet(num_ch=3,num_res=32, num_feat=128).to(device)
    print(f'Generator Parameters Size:{sum(p.numel() for p in netG.parameters() if p.requires_grad) / 1000000.0 :.2f} MB')
    #创建判别器
    netD = Discriminator(in_channels = 3).to(device)
    print(f'Discriminator Parameters Size:{sum(p.numel() for p in netD.parameters() if p.requires_grad) / 1000000.0 :.2f} MB')
    
    # 加载预训练模型
    if(checkpoint_path != None and checkpoint_path != ''):
        print("加载检查点...")
        checkpoint = torch.load(checkpoint_path,map_location=torch.device(device=device))
        netG.load_state_dict(checkpoint['model_state_dict'])
        netD.load_state_dict(checkpoint['D_model_state_dict'])
        print("载检查点已加载!!!")

    # 创建生成器损失函数对象GeneratorLoss
    generator_criterion = GeneratorLoss().to(device)
    #ssim计算-pytorch.ssim亲测不好用
    ssim = SSIM().to(device=device)

    #构造迭代器
    optimizerG = optim.Adam(netG.parameters(), lr=0.001)
    optimizerD = optim.Adam(netD.parameters(), lr=0.001)
    #学习率衰减策略
    lf = lambda x:((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.00001) + 0.00001
    schedulerG = lr_scheduler.LambdaLR(optimizerG, lr_lambda=lf)
    schedulerD = lr_scheduler.LambdaLR(optimizerD, lr_lambda=lf)

    # 创建一个字典用于存储训练过程中的判别器和生成器的损失、分数和评估指标结果(信噪比和相似性)
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    print("-----------------------初始化完成！！！开始训练！！！-----------------------")
    for epoch in range(1, epochs + 1):
        # 创建训练数据的进度条
        start = time.perf_counter()
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        netG.train()  # 将生成器设置为训练模式
        netD.train()  # 将判别器设置为训练模式
        for LR_img, HR_img in train_bar:
            LR_img, HR_img = LR_img.to(device), HR_img.to(device)
            batch_size = LR_img.size(0)
            running_results['batch_sizes'] += batch_size

            fake_img = netG(LR_img)

            # 清除判别器的梯度
            netD.zero_grad()
            # 通过判别器对真实图像进行前向传播，并计算其输出的平均值
            real_out = netD(HR_img).mean()
            # 通过判别器对伪图像进行前向传播，并计算其输出的平均值
            fake_out = netD(fake_img).mean()
            # 计算判别器的损失
            d_loss = 1 - real_out + fake_out
            # 在判别器网络中进行反向传播，并保留计算图以进行后续优化步骤
            d_loss.backward(retain_graph=True)
            # 利用优化器对判别器网络的参数进行更新
            optimizerD.step()

            netG.zero_grad()
            # The two lines below are added to prevent runtime error in Google Colab
            # 通过生成器对输入图像（z）进行生成，生成伪图像（fake_img）
            fake_img = netG(LR_img)
            # 通过判别器对伪图像进行前向传播，并计算其输出的平均值
            fake_out = netD(fake_img).mean()
            # 计算生成器的损失，包括对抗损失、感知损失、图像损失和TV损失
            g_loss = generator_criterion(fake_out, fake_img, HR_img)
            # 在生成器网络中进行反向传播，计算生成器的梯度
            g_loss.backward()

            # 再次通过生成器对输入图像（z）进行生成，得到新的伪图像（fake_img）
            fake_img = netG(LR_img)
            # 通过判别器对新的伪图像进行前向传播，并计算其输出的平均值
            fake_out = netD(fake_img).mean()
            # 利用优化器对生成器网络的参数进行更新
            optimizerG.step()

            # 累加当前批次生成器的损失值乘以批次大小，用于计算平均损失
            running_results['g_loss'] += g_loss.item() * batch_size
            # 累加当前批次判别器的损失值乘以批次大小，用于计算平均损失
            running_results['d_loss'] += d_loss.item() * batch_size
            # 累加当前批次真实图像在判别器的输出得分乘以批次大小，用于计算平均得分
            running_results['d_score'] += real_out.item() * batch_size
            # 累加当前批次伪图像在判别器的输出得分乘以批次大小，用于计算平均得分
            running_results['g_score'] += fake_out.item() * batch_size
            # 更新训练进度条的描述信息
            train_bar.set_description(desc='[train epoch-%d/%d] Loss_D: %.4f Loss_G: %.4f Score_D: %.4f Score_G: %.4f' % (
                epoch, epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
        #一轮训练结束
        end = time.perf_counter()
        print(f"-----------------------第{epoch}轮训练的时长为:{(end - start):.2f}s,开始验证！-----------------------")
        #开始验证本轮
        netG.eval()
        with torch.no_grad():
            val_bar = tqdm(valid_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr in val_bar:
                val_lr, val_hr = val_lr.to(device), val_hr.to(device)
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size

                val_fake = netG(val_lr)

                # 计算批量图像的均方误差
                batch_mse = ((val_fake - val_hr) ** 2).data.mean()
                # 累加均方误差
                valing_results['mse'] += batch_mse * batch_size
                # 计算批量图像的结构相似度指数
                batch_ssim = ssim(val_fake, val_hr).item()
                # 累加结构相似度指数
                valing_results['ssims'] += batch_ssim * batch_size
                # 计算平均峰值信噪比
                valing_results['psnr'] = 10 * log10(
                    (val_hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                # 计算平均结构相似度指数
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                # 更新训练进度条的描述信息
                val_bar.set_description(
                    desc='[valid epoch-%d] PSNR: %.4f dB, SSIM: %.4f, lr: %f' % (
                        epoch, valing_results['psnr'], valing_results['ssim'], optimizerG.state_dict()['param_groups'][0]['lr']))
                dump_frist_output(val_fake,path="temp/output.jpg")
                dump_frist_output(val_hr,path="temp/target.jpg")
                dump_frist_output(val_lr,path="temp/input.jpg")
            #学习率更新
            schedulerG.step()
            schedulerD.step()

            # 将判别器和生成器的参数保存到指定文件
            torch.save({
                "model_state_dict": netG.state_dict(),
                "D_model_state_dict": netD.state_dict()
            }, os.path.join(checkpoints_save_folder,f'checkpoint_GAN_{epoch}.pth'))

            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])

    out_path = './statistics'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 创建一个DataFrame对象，用于存储训练结果数据
    data_frame = pd.DataFrame(
        data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
              'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        index=range(1, epoch + 1))
    # 将DataFrame对象保存为CSV文件
    data_frame.to_csv(out_path + '/train_results.csv', index_label='Epoch')