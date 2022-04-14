import torchvision
from PIL import Image
from model import *
img_path = r"E:\train\man\001.jpg"
image = Image.open(img_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

image = transform(image)
print(image)

# 加载网络模型
model = torch.load("xjl_nn_19.pth")  # 在这里替换模型
# print(model)
# 给图片添加batch_size
image = torch.reshape(image, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    output = model(image)
print(output.argmax(1))
