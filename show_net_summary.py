# 网络结构
from nets.facenet import facenet

if __name__ == "__main__":
    input_shape = [160, 160, 3]
    model = facenet(input_shape, 10575, mode="train")
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i, layer.name)
