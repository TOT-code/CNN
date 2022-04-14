from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure, color
import matplotlib.pyplot as plt
from skimage import io

path = r"E:\2D_Data\NewProcessData\bus\056.jpg"
img = resize(color.rgb2gray(io.imread(path)), (120, 60))
resized_img = resize(img, (40, 20))
fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=False)

"""
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()
"""
