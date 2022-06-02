
from keras import backend as K
from keras.layers import Activation, Conv2D, Dense, DepthwiseConv2D, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


# 普通卷积块: 标准的卷积+标准化+激活函数
def Conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


# 可分离卷积块: 利用更少的参数代替普通3x3卷积
def Depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    # 3x3可分离卷积
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # 1x1普通卷积
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    return K.relu(x, max_value=6)


def JinNet(inputs, embedding_size=128, dropout_keep_prob=0.4, alpha=1.0, depth_multiplier=1):
    # 160,160,3 -> 80,80,32， 压缩像素， 拉长通道
    x = Conv_block(inputs, 32, strides=(2, 2))
    
    # 80,80,32 -> 80,80,64， 默认步长(1,1)
    x = Depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 80,80,64 -> 40,40,128
    x = Depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    x = Depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 40,40,128 -> 20,20,256
    x = Depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    x = Depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 20,20,256 -> 10,10,512
    x = Depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = Depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 10,10,512 -> 5,5,1024
    x = Depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = Depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)
    
    # 5,5,1024 -> 128特征向量
    # 1024
    x = GlobalAveragePooling2D()(x)
    # Dropout层
    # 防止网络过拟合，训练的时候起作用
    # dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
    # 注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。
    # dropout是CNN中防止过拟合提高效果的一个大杀器，但对于其为何有效，却众说纷纭。
    # https://blog.csdn.net/qq_34216467/article/details/83141837?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165015597016782248524001%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165015597016782248524001&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-83141837.142^v9^pc_search_result_control_group,157^v4^control&utm_term=dropout防止过拟合&spm=1018.2226.3001.4187
    # https://blog.csdn.net/zhaoxr233/article/details/90653274?ops_request_misc=&request_id=&biz_id=102&utm_term=Dropout(Layer)&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-90653274.142^v9^pc_search_result_control_group,157^v4^control&spm=1018.2226.3001.4187
    x = Dropout(1.0 - dropout_keep_prob, name='Dropout')(x)
    # 全连接层到128
    x = Dense(embedding_size, use_bias=False, name='Bottleneck')(x)
    # 你再也不用去理会过拟合中drop out、L2正则项参数的选择问题，采用BN算法后，你可以移除这两项了参数，或者可以选择更小的L2正则约束参数了
    # BN层是对于每个神经元做归一化处理，甚至只需要对某一个神经元进行归一化，而不是对一整层网络的神经元进行归一化。
    # 既然BN是对单个神经元的运算，那么在CNN中卷积层上要怎么搞？
    # 假如某一层卷积层有6个特征图，每个特征图的大小是100*100，这样就相当于这一层网络有6*100*100个神经元，
    # 如果采用BN，就会有6*100*100个参数γ、β，这样岂不是太恐怖了。因此卷积层上的BN使用，其实也是使用了类似权值共享的策略，把一整张特征图当做一个神经元进行处理
    # https://blog.csdn.net/hjimce/article/details/50866313?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165015521616780274125067%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=165015521616780274125067&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-50866313.142^v9^pc_search_result_control_group,157^v4^control&utm_term=BatchNormalization&spm=1018.2226.3001.4187
    x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='BatchNorm_Bottleneck')(x)
 
    # 建模型
    model = Model(inputs, x, name='mobilenet')

    return model
