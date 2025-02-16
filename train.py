import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

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
            dump_frist_output(inputs,path="input.png")
            dump_frist_output(outputs,path="output.png")
            dump_frist_output(targets,path="target.png")
        print('Batch %d loss: %.4f'%(batch_idx,loss.item()))

    print(f"Training Loss: {running_loss/len(dataloader):.4f}")

if __name__ == '__main__':
    # 设置参数
    image_folder = "data/DIV2K_train_HR"
    model = SRSCNet()  # 确保你的模型支持单通道输入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道，均值和标准差只需一个数值
    ])

    dataset = HDImageDataset(image_folder, transform=transform,crop_size=(512,512))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    criterion = torch.nn.L1Loss()  # 可以尝试其他损失函数，如MSELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100):  # 训练10个epoch作为例子
        print(f"Epoch {epoch+1}/100")
        train(model, dataloader, criterion, optimizer, device)
        if(epoch%10 == 9):
            save_checkpoint(model, optimizer, epoch, 0)