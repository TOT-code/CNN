# 网络模型，数据，损失函数+cuda

import datetime

import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, Dropout, ReLU
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# syntax 2  会根据设备情况自动选择cpu或者gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备数据集
# BATCH_SIZE = 4
BATCH_SIZE = 16
use_gpu = torch.cuda.is_available()
transform_train = transforms.Compose([
    transforms.Resize([120, 60]),
    # transforms.Resize([40, 40]),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
])
transform_test = transforms.Compose([
    transforms.Resize([120, 60]),
    # transforms.Resize([40, 40]),
    # transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
])

# 读取数据
dataset_train = datasets.ImageFolder('/content/CNN/data/train', transform_train)
dataset_test_1 = datasets.ImageFolder('/content/CNN/data/Small_data/test', transform_test)
dataset_test_2 = datasets.ImageFolder('/content/CNN/data/Small_data/train', transform_test)
dataset_test_3 = datasets.ImageFolder('/content/CNN/data/test', transform_test)
dataset_test = dataset_test_1 + dataset_test_2+dataset_test_3
# 获取数据集长度
dataset_train_size = len(dataset_train)
dataset_test_size = len(dataset_test)
print("训练数据集长度：{}".format(dataset_train_size))
print("测试数据集长度：{}".format(dataset_test_size))

# Data_loader加载
Data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)  # 训练集
Data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)  # 测试集
# 搭建神经网络（一般要独立设置为一个脚本文件,这里是model.py文件
# 创建网络模型
# xjl_nn = XJL()
xjl_nn = torchvision.models.vgg16(pretrained=False)
xjl_nn.classifier= torch.nn.Sequential(
    Linear(in_features=25088, out_features=4096, bias=True),
    ReLU(inplace=True),
     Dropout(p=0.5, inplace=False),
    Linear(in_features=4096, out_features=4096, bias=True),
    ReLU(inplace=True),
    Dropout(p=0.5, inplace=False),
    Linear(in_features=4096, out_features=1000, bias=True),
    )  # 删除Dropout
xjl_nn.add_module('add_linear', nn.Linear(1000, 96))
xjl_nn.add_module('add_linear', nn.Linear(96, 3))
xjl_nn = xjl_nn.cuda()
print(xjl_nn)
# 损失函数
loss = nn.CrossEntropyLoss().cuda()

# 优化器
learn_rate = 1e-3
optimizer = torch.optim.SGD(xjl_nn.parameters(), lr=learn_rate)

# 设置训练网络参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 100
# 添加tensorboard
log_dir = "logs_fit_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
writer = SummaryWriter(log_dir)

for i in range(epoch):
    print("----第 {} 轮训练开始----".format(i + 1))

    # 训练步骤开始
    xjl_nn.train()  # 有dropout和normlizer层需要调用
    for data in Data_loader_train:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        output_x = xjl_nn(imgs)
        loss_x = loss(output_x, targets)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清0
        loss_x.backward()  # 求解grad梯度
        optimizer.step()  # 更新weight

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{} ".format(total_train_step, loss_x.item()))
            writer.add_scalar("train_loss", loss_x.item(), total_train_step)

    # 测试步骤开始
    xjl_nn.eval()  # 同理
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in Data_loader_test:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output_x = xjl_nn(imgs)
            loss_x = loss(output_x, targets)
            total_test_loss = total_test_loss + loss_x.item()
            accuracy = (output_x.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / dataset_test_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / dataset_test_size, total_test_step)
    total_test_step = total_test_step + 1
    if i == epoch - 1:
        torch.save(xjl_nn, "xjl_nn_{}.pth".format(i))
    print("模型已保存")

writer.close()
