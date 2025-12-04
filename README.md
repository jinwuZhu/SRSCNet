# SRSCNet

SRSCNet是一个单卷积图像超分轻量化模型

<table>
  <tr>
    <td><img src="temp/input.jpg" width="256px"></td>
    <td><img src="images/right.png" width="100"></td>
    <td><img src="temp/output.jpg" width="256px"></td>
  </tr>
</table>

### 特性

- 1x1 卷积网络，实现轻量级图像超分辨率。
- 该模型可以通过极小的参数量训练产生不错的图像增强效果

### 支持的功能
- 图片2倍超分，增强

### 效果展示

- 图像超清对比（左边是普通放大方式，右边是SRSCNet）

<table>
<tr><td><img src="images/comic_lsr.jpg"></td><td><img src="images/comic_sr.jpg"></td></tr>
<tr><td><img src="images/butterfly_GT_lsr.jpg"></td><td><img src="images/butterfly_GT_sr.jpg"></td></tr>
<table>

### 如何使用
- 使用预训练的模型: 
```shell
python enhance.py -i 'input.jpg' -o 'sr_image.jpg' -m 'checkpoints/checkpoint_epoch_1_batch_0.pth'
```

- 完整命令参数请使用help命令：

```shell
>> python ./enhance.py -h
usage: enhance.py [-h] [--model MODEL] [--input INPUT] [--output OUTPUT]

图像增强

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        模型检查点路径
  --input INPUT, -i INPUT
                        输入文件路径
  --output OUTPUT, -o OUTPUT
                        输出文件路径 (默认: sr_image.jpg)
```

### 安装
1. 你需要安装python3.9 + 以及pytroch,numpy,cv2等依赖库
```
pip install torch torchvision
pip install numpy
pip install opencv-python
```

2.  获取训练好的模型，并放到工程目录下。（注意：你也可以另外放到指定目录下，但是你需要指定 --model 参数来确定加载地址）

### 如何训练
本工程的训练均是由高清图下采样得到的数据集，你只需要准备一个文件夹存放图片数据即可

- 非GAN的训练脚本 <b>train.py</b>

```
python train.py --datafolder '图片文件夹路径'
```

- GAN对抗训练

```
python traingan.py --datafolder '图片文件夹路径' --device 'cpu'
```

### 训练建议

- <b>初期:</b> 

如果只有高清图像进行训练（目前内置训练脚本），建议在训练初期，对于输入，使用2次下采样后再上采样一次，这样能更快的训练出轮廓。之后再进行2x的训练进行微调（主要目的是消除前期4x的伪影，同时优化细节）。

- <b>中后期:</b>

批次大小可以缓慢降低，输入输出尺寸也可以尝试缓慢降低。

### 引用
- [Fully 1×1 Convolutional Network for Lightweight Image Super-Resolution](http://arxiv.org/abs/2307.16140)

