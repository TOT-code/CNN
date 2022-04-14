"""
对较大的1维数据集进行训练
"""
# 1D-CNN

from model_2 import *
import numpy
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import datetime


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, idx):
        txt_name = self.data_path[idx]
        data_item_path = os.path.join(self.root_dir, self.label_dir, txt_name)
        data = numpy.loadtxt(data_item_path)

        # list = []
        # data = list.append(data)

        label = self.label_dir
        return data, label

    def __len__(self):
        return len(self.data_path)


train_data_dir = r'E:\1D-Data\Big_Data\train'
test_data_dir = r'E:\1D-Data\Small_Data\val'
bus_label_dir = '0'
car_label_dir = '1'
man_label_dir = '2'

train_bus_data = MyData(train_data_dir, bus_label_dir)
train_car_data = MyData(train_data_dir, car_label_dir)
train_man_data = MyData(train_data_dir, man_label_dir)
train_data = train_bus_data + train_car_data + train_man_data
test_bus_data = MyData(test_data_dir, bus_label_dir)
test_car_data = MyData(test_data_dir, car_label_dir)
test_man_data = MyData(test_data_dir, man_label_dir)
test_data = test_bus_data + test_car_data + test_man_data

BatchSize = 16

Data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=BatchSize, shuffle=True, drop_last=True)  # 训练集
Data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=BatchSize, shuffle=True, drop_last=True)  # 测试集

# 获取数据集长度
dataset_train_size = len(Data_loader_train)
dataset_test_size = len(Data_loader_test)
print("训练数据集长度：{}".format(dataset_train_size))
print("测试数据集长度：{}".format(dataset_test_size))
OneD_nn = Net()

# 损失函数
loss = nn.CrossEntropyLoss()

# 优化器
learn_rate = 1e-2
optimizer = torch.optim.SGD(OneD_nn.parameters(), lr=learn_rate)

# 设置训练网络参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 50
# 添加tensorboard
log_dir = "logs_fit_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
writer = SummaryWriter(log_dir)

for i in range(epoch):
    print("----第 {} 轮训练开始----".format(i + 1))

    # 训练步骤开始
    OneD_nn.train()  # 有dropout和normlizer层需要调用
    for data_batch in Data_loader_train:
        data, targets = data_batch
        empty_list_1 = []
        for j in range(BatchSize):
            # print(data_1[i].size())
            # print(data_1[i])
            b = data[j].numpy()
            b = [b.tolist()]
            empty_list_1.append(b)
        empty_list_1 = torch.Tensor(empty_list_1)

        list_target_1 = []
        for j in range(BatchSize):
            list_target_1.append(int(targets[j]))
        list_target_1 = torch.LongTensor(list_target_1)

        '''
        if targets == ('bus',):
            targets = [0]
        elif targets == ('car',):
            targets = [1]
        elif targets == ('man',):
            targets = [2] 
        '''
        # targets = torch.LongTensor(targets)
        # print(datas)
        output_x = OneD_nn(empty_list_1)
        # print(list_target)
        # print(output_x)
        loss_x = loss(output_x, list_target_1)

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清0
        loss_x.backward()  # 求解grad梯度
        optimizer.step()  # 更新weight

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{} ".format(total_train_step, loss_x.item()))
            writer.add_scalar("train_loss", loss_x.item(), total_train_step)

    # 测试步骤开始
    OneD_nn.eval()  # 同理
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data_batch in Data_loader_test:
            data, targets = data_batch

            empty_list_2 = []
            for j in range(BatchSize):
                # print(data_1[i].size())
                # print(data_1[i])
                b = data[j].numpy()
                b = [b.tolist()]
                empty_list_2.append(b)
            empty_list_2 = torch.Tensor(empty_list_2)

            list_target_2 = []
            for j in range(BatchSize):
                list_target_2.append(int(targets[j]))
            list_target_2 = torch.LongTensor(list_target_2)
            '''
            if targets == ('bus',):
                targets = [0]
            elif targets == ('car',):
                targets = [1]
            elif targets == ('man',):
                targets = [2] 
            '''
            # targets = torch.LongTensor(targets)

            output_x = OneD_nn(empty_list_2)
            loss_x = loss(output_x, list_target_2)
            total_test_loss = total_test_loss + loss_x.item()
            accuracy = (output_x.argmax(1) == list_target_2).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print(total_accuracy, '\n', (dataset_test_size * BatchSize))
    print("整体测试集上的正确率：{}".format(total_accuracy / (dataset_test_size * BatchSize)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / (dataset_test_size * BatchSize), total_test_step)
    total_test_step = total_test_step + 1
    if i == epoch - 1:
        torch.save(OneD_nn, "OneD_nn_{}.pth".format(i))
    # torch.save(xjl_nn.state_dict(),"xjl_nn_{}.pth".format(i))
    print("模型已保存")

writer.close()
