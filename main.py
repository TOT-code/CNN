from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.image_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.image_path)


root_dir = r'E:\train'
man_label_dir = 'man'
man_dataset = MyData(root_dir, man_label_dir)
bus_label_dir = 'bus'
bus_dataset = MyData(root_dir,bus_label_dir)
car_label_dir = 'car'
car_dataset = MyData(root_dir,car_label_dir)
train_data = man_dataset + bus_dataset + car_dataset
idx = 0
img, label = car_dataset[idx]



