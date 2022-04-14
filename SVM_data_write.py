from skimage.transform import resize
from skimage.feature import hog
from skimage import color
from skimage import io
import os

from torch.utils.data import Dataset


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename, 'a')
    for i in range(len(data)):
        file.write(str(data[i]))
        if i < (len(data) - 1):
            file.write(',')
    file.write('\n')
    file.close()
    print("保存文件成功")


class ImgDir(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.data_path = os.listdir(self.path)

    def __getitem__(self, idx):
        txt_name = self.data_path[idx]
        data_item_path = os.path.join(self.root_dir, self.label_dir, txt_name)

        return data_item_path

    def len(self):
        return len(self.data_path)


data_dir = r'E:\2D_Data\Small_data\test'
label_dir = 'man'
bus_data = ImgDir(data_dir, label_dir)
length = bus_data.len()

for i in range(length):
    path = (bus_data[i])
    img = resize(color.rgb2gray(io.imread(path)), (120, 60))
    resized_img = resize(img, (96, 48))
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=False)
    fd = fd.tolist()
    fd.append(label_dir)
    file_name = 'SVM.data'
    text_save(file_name, fd)
