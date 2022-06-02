import keras.backend as K
from keras.layers import Activation, Dense, Input, Lambda
from keras.models import Model
from nets.jinnet import JinNet


def facenet(input_shape, num_classes = None, mode = "train"):
    inputs = Input(shape=input_shape)
    # 利用主干网络进行特征提取
    model = JinNet(inputs, dropout_keep_prob = 0.5)

    if mode == "train":
        # 训练的话利用交叉熵和triplet_loss结合一起训练
        # 全连接完成交叉熵分类器， num_classes分类个数
        logits = Dense(num_classes)(model.output)

        # 因为进行了交叉熵的计算， 所以添加softmax层
        # softmax loss = softmax + cross_entropy loss
        softmax = Activation("softmax", name = "Softmax")(logits)
        
        # 用于triplet loss的构建
        # Lambda 将任意表达式封装为 Layer 对象
        normalize = Lambda(lambda  x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        combine_model = Model(inputs, [softmax, normalize])

        return combine_model

    elif mode == "predict":
        # 预测的时候只需要考虑人脸的特征向量就行了
        normalize = Lambda(lambda  x: K.l2_normalize(x, axis=1), name="Embedding")(model.output)
        model = Model(inputs,normalize)
        
        return model
