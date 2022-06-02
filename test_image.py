from PIL import Image
import cv2
from utils.utils import image_normalization, resize_image
import numpy as np

# image_1 = Image.open('img/3.jpg')
image = cv2.imread('img/JinShaoFei.jpg')

photo = np.expand_dims(image_normalization(np.array(image, np.float32)), 0)
# print(image_1.size)
print(photo.shape)
