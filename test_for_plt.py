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
        "model_path"        : "model_data/facenet_mobilenet.h5",
        "input_shape"       : [160, 160, 3],
    }

    @classmethod
    def get_initialAttr(cls, n):
        if n in cls.intitial_attributions:
            return cls.intitial_attributionss[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self.intitial_attributions)
        # 允许添加额外的参数
        for name, value in kwargs.items():
            # 设置属性， self.name = value
            setattr(self, name, value)
            
        self.load()

    def load(self): 
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        self.model = facenet(self.input_shape, mode="predict")
        
        print('正在载入模型...')
        self.model.load_weights(self.model_path, by_name=True)
        print('{} , 模型已加载完毕.'.format(model_path))

    def detect_image(self):
        image_1 = Image.open('test_1.jpg')
        image_2 = Image.open('test_2.jpg')

        image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]])
        image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]])
        
        photo_1 = np.expand_dims(image_normalization(np.array(image_1, np.float32)), 0)
        photo_2 = np.expand_dims(image_normalization(np.array(image_2, np.float32)), 0)

        output_1 = self.model.predict(photo_1)
        output_2 = self.model.predict(photo_2)

        loss = np.linalg.norm(output_1-output_2, axis=1)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % loss, ha='center', va= 'bottom',fontsize=11)
        plt.show()

        return loss

if __name__ == '__main__':
    
    model = Facenet()
    model.detect_image()
