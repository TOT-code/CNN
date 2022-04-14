import numpy
import torch

from model_2 import *

txt_path = r"E:\1D-Data\val\1\00211.txt"
data = numpy.loadtxt(txt_path)

empty_list = []
data_1 = torch.from_numpy(data)
b = data_1.numpy()
b = [b.tolist()]
empty_list.append(b)
empty_list = torch.Tensor(empty_list)

# 加载网络模型
model = torch.load("OneD_nn_9.pth")  # 在这里替换模型
# print(model)

model.eval()
with torch.no_grad():
    output = model(empty_list)
print(output.argmax(1))
