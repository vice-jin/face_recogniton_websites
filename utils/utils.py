import cv2
import os
import numpy as np
from PIL import Image
from scipy.misc import imsave
from flask import flash


# 检查如果文件的扩展名是一个允许的扩展名， 那么就上传
# filename: 待查看的上传文件的文件名。
# allowed_set: 有效的文件扩展名的集合。
def allowed_file(filename, allowed_set):    
    # Returns:
    #     check: 布尔值表示文件扩展名是否在允许的扩展名列表中
    #             True = 允
    #             False = 杀

    # rsplit(分隔符， 分隔次数)
    check = '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

    return check


def remove_file_extension(filename):
    # 返回不带文件扩展名的图像文件名，用于文件存储。
    filename = os.path.splitext(filename)[0]

    return filename


def save_image(img, filename, uploads_path):
    # 保存一个图像文件到'uploads'文件夹。
    try:
        path = os.path.join(uploads_path, filename)
        imsave(path, arr=np.squeeze(img))
        flash("Image saved!")
    except Exception as e:
        print(str(e))

        return str(e)

    return path


# 将图像转换成RGB图像，防止灰度图在预测时报错。
# 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


# 对输入图像进行resize, 缩小到160以下， 填充边框
# size = (weight, height)
def resize_image(image, size, letterbox_image=True):
    iw, ih  = image.size
    # iw, ih, _ = image.shape
    w, h    = size
    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    # image = cv2.resize(image, (nw, nh))
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image


# 将导出的labels和image地址分开， 提取其中第一段的labels
def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


# 图片归一化处理
def image_normalization(image):
    image /= 255.0 
    return image


if __name__ == '__main__':
    img = Image.open('test_1.jpg')
    resize_image(img, (160, 160))
    print(save_image(img, 'test_1', '/User/img'))