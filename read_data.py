import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

BATCH_SIZE = 4
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
])

# 读取数据
dataset_train = datasets.ImageFolder(r'E:\train', transform_train)
dataset_test = datasets.ImageFolder(r'E:\test', transform_test)
# 上面这一段是加载测试集的
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)  # 训练集
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)  # 测试集
# 对应文件夹的label
print(dataset_train.class_to_idx)  # 这是一个字典，可以查看每个标签对应的文件夹，也就是你的类别。
# 训练好模型后输入一张图片测试，比如输出是99，就可以用字典查询找到你的类别名称
print(dataset_test.class_to_idx)
# print(dataset_val.class_to_idx)
