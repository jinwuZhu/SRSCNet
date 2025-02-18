import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from argparse import ArgumentParser

from models import SRSCNet
from datasets import HDImageDataset

def save_checkpoint(model, optimizer, epoch, batch_idx, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'batch_idx': batch_idx
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints', epoch=None, batch_idx=None):
    """
    加载检查点。
    
    参数:
    - model: 要恢复状态的模型。
    - optimizer: 要恢复状态的优化器。
    - checkpoint_dir: 检查点目录，默认为'checkpoints'。
    - epoch: 如果指定，则尝试加载特定epoch的检查点；否则加载最新的检查点。
    - batch_idx: 如果指定，则与epoch一起用于定位特定的检查点。
    
    返回:
    - epoch: 从检查点中恢复的epoch。
    - batch_idx: 从检查点中恢复的batch索引。
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist.")
        return None, None
    
    # 如果没有指定epoch，查找最近的检查点
    if epoch is None:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not checkpoints:
            print("No checkpoints found.")
            return None, None
        
        def extract_numbers(name):
            parts = name.split('_')
            epoch_part = int(parts[2])
            batch_part = int(parts[4].split('.')[0]) if len(parts) > 4 else 0
            return epoch_part, batch_part
        
        latest_checkpoint = max(checkpoints, key=extract_numbers)
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    else:
        # 尝试根据提供的epoch和batch_idx加载特定的检查点
        if batch_idx is None:
            checkpoint_filename = f'checkpoint_epoch_{epoch}_batch_0.pth'
        else:
            checkpoint_filename = f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return None, None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint['epoch'], checkpoint['batch_idx']

from utils import dump_frist_output
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if(batch_idx == 0):
            dump_frist_output(inputs,path="temp/input.png")
            dump_frist_output(outputs,path="temp/output.png")
            dump_frist_output(targets,path="temp/target.png")
        print('Batch %d loss: %.4f'%(batch_idx,loss.item()))

    print(f"Training Loss: {running_loss/len(dataloader):.4f}")

if __name__ == '__main__':
    parser = ArgumentParser(description='图像增强')
    parser.add_argument('--checkpoint','-c', type=str,default='no', help='从检查点继续？yes:no')
    parser.add_argument('--datafolder', '-d', type=str, default='data/DIV2K_train_HR', help='训练数据集路径')

    args = parser.parse_args()

    # 如果需要从检查点继续训练，将此值设置为True
    continue_checkpoint = True if args.datafolder == "yes" else False
    # 设置参数
    image_folder = args.datafolder
    model = SRSCNet(num_ch=1,num_res=16,num_feat=32)  # 确保你的模型支持单通道输入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道，均值和标准差只需一个数值
    ])

    dataset = HDImageDataset(image_folder, transform=transform,crop_size=(256,256))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    criterion = torch.nn.L1Loss()  # 可以尝试其他损失函数，如MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 如果需要从检查点继续
    if(continue_checkpoint == True):
        load_checkpoint(model=model,optimizer=optimizer)

    for epoch in range(100):  # 训练10个epoch作为例子
        print(f"Epoch {epoch+1}/100")
        train(model, dataloader, criterion, optimizer, device)
        if(epoch%10 == 9):
            save_checkpoint(model, optimizer, epoch, 0)