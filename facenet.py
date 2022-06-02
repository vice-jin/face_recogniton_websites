import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nets.facenet import facenet
from utils.utils import image_normalization, resize_image


class Facenet(object):
    intitial_attributions = {
        # 权重文件
        # 验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        "model_path"        : "model_data/facenet_mobilenet.h5",

        # 输入图片的大小。
        "input_shape"       : [160, 160, 3],
    }


    # 采用classmethod修饰符的方式，这样定义出来的函数就能够在类对象实例化之前调用这些函数
    # 使用@classmethod，不创建实例就能调用此类方法
    @classmethod
    def get_initialAttr(cls, n):
        if n in cls.intitial_attributions:
            return cls.intitial_attributionss[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    # 当访问一个对象的会根据不同的情况作不同的处理，是比较复杂的。
    # 一般象a.b这样的形式，python可能会先查找a.__dict__中是否存在，如果不存在会在类的__dict__中去查找，再没找到可能会去按这种方法去父类中进行查找
    # 初始化Facenet
    def __init__(self, **kwargs):
        self.__dict__.update(self.intitial_attributions)
        # 允许添加额外的参数
        for name, value in kwargs.items():
            # 设置属性， self.name = value
            setattr(self, name, value)
            
        self.load()
        

    # 载入模型
    def load(self): 
        # 载入模型与权值
        # os.path.expanduser: 可以在路径上+'~/'
        model_path = os.path.expanduser(self.model_path)
        # assert若为true跳过， false执行后面语句
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model = facenet(self.input_shape, mode="predict")
        
        print('正在载入模型...')
        # 加载权重文件
        self.model.load_weights(self.model_path, by_name=True)
        print('{} , 模型已加载完毕.'.format(model_path))
    

    # 计算出image的128的特征向量
    def load_model_vector(self, image):
        # 保证图片尺寸符合神经网络的输入格式
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]])
            
        # 图片归一化
        photo = np.expand_dims(image_normalization(np.array(image, np.float32)), 0)

        # 图片传入网络进行预测
        output = self.model.predict(photo)

        return output


    # 保存图片的128特征向,保存在 'loss' 下, 以 filename.txt 保存
    # image: Image图片
    # filename: 保存loss的名字， 即预测的名字
    # path ： loss保存路径
    def save_model_vector(self, image, filename, path='loss'):
        filename = os.path.splitext(filename)[0] + '.txt'
        # 保证图片尺寸符合神经网络的输入格式
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]])
            
        # 图片归一化
        # (1, x, y, channel)
        photo = np.expand_dims(image_normalization(np.array(image, np.float32)), 0)

        # 图片传入网络进行预测
        output1 = self.model.predict(photo)

        full_path = os.path.join(path, filename)
        print('loss function save to :', full_path)

        # 保存128特征向量到文件中
        np.savetxt(full_path, output1)

        # list_file = open('test_for_loss.txt', 'w')
        # list_file.write(str(output1))
        # list_file.write('\n')    


    # 检测图片
    def detect_image(self, image_1, image_2):
        # 保证图片尺寸符合神经网络的输入格式
        image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]])
        image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]])
        
        # 图片归一化
        photo_1 = np.expand_dims(image_normalization(np.array(image_1, np.float32)), 0)
        photo_2 = np.expand_dims(image_normalization(np.array(image_2, np.float32)), 0)


        # 图片传入网络进行预测
        output_1 = self.model.predict(photo_1)
        output_2 = self.model.predict(photo_2)

        # 计算二者之间的距离
        loss = np.linalg.norm(output_1-output_2, axis=1)

        # matlab输出
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))

        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Distance:%.3f' % loss, ha='center', va= 'bottom',fontsize=11)
        # plt.show()

        return loss


if __name__ == '__main__':
    image = Image.open('img/sl.jpg')

    model = Facenet()
    # model.save_model_vector(image, filename='jsf')
    print(model.save_model_vector(image, 'sl'))