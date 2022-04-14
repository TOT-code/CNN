import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout, Softmax


# 搭建神经网络（一般要独立设置为一个脚本文件
class XJL(nn.Module):
    def __init__(self):
        super(XJL, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, (5, 5), padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Dropout(p=0.5),
            Linear(64, 3),
            ReLU()
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    xjl_nn = XJL()
    input_x = torch.ones((64, 3, 32, 32))
    output_x = xjl_nn(input_x)
    print(output_x)

# 一维神经网络model
'''
            Conv1d(1, 16, kernel_size=11),
            Conv1d(16, 16, kernel_size=3),
            Conv1d(16, 16, kernel_size=3),
            MaxPool1d(3),
            Conv1d(16, 64, kernel_size=3),
            Conv1d(64, 64, kernel_size=3),
            MaxPool1d(3),
            Conv1d(64, 64, kernel_size=3),
            Conv1d(64, 64, kernel_size=3),
            MaxPool1d(3),
            Flatten(),
            Linear(448, 64),
            Linear(64, 3)
'''
# 二维神经网络
"""
Conv2d(3, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, (5, 5), padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, (5, 5), padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Dropout(p=0.5),
            Linear(64, 3),
            ReLU()
"""
