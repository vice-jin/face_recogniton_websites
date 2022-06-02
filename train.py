import datetime
import os
import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, Dense, DepthwiseConv2D, PReLU
from keras.optimizers import SGD
from keras.regularizers import l2
from nets.facenet import facenet
from nets.facenet_training import get_lr_scheduler, triplet_loss
from utils.callbacks import LFW_callback, LossHistory
from utils.dataloader import Dataset, LFWDataset
from utils.utils import get_num_classes


if __name__ == "__main__":
    
    # 指向根目录下的cls_train.txt，读取人脸路径与标签
    txt_path = "train.txt"

    # 默认输入图片的大小， [160, 160, 3]或者[112, 112, 3]
    input_shape = [160, 160, 3]
    
    # batch_size      每次输入的图片数量， 受到数据加载方式与triplet loss的影响， batch_size需要为3的倍数。
    # Epoch           模型总共训练的epoch
    batch_size  = 96
    Epoch = 100

    # save_dir        权值与日志文件保存的文件夹
    save_dir = 'logs'
    
    # 提取labels
    num_classes = get_num_classes(txt_path)

    # 载入模型并加载预训练权重
    model = facenet(input_shape, num_classes, mode="train")
    # 载入预训练权重
    model.load_weights("model_data/facenet_mobilenet.h5", by_name=True, skip_mismatch=True)
    # 0.3用于验证，0.7用于训练
    val_split = 0.3

    with open(txt_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101) # 随机数
    np.random.shuffle(lines) # 打乱
    np.random.seed(None)

    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # for layer in model.layers:
    #     # 判断layer是哪一层
    #     if isinstance(layer, DepthwiseConv2D):
    #         layer.add_loss(l2(0)(layer.depthwise_kernel))
    #     elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
    #         layer.add_loss(l2(0)(layer.kernel))
    #     elif isinstance(layer, PReLU):
    #         layer.add_loss(l2()(layer.alpha))
            
    if True:
        if batch_size % 3 != 0:
            raise ValueError("3的倍数")
        # 判断当前batch_size，自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-1
        lr_limit_min = 5e-4

        # Init_lr:模型的最大学习率, 用于学习率调节器
        # Min_lr:模型的最小学习率，默认为最大学习率的0.01
        Init_lr = 1e-2
        Min_lr = Init_lr * 0.01
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = SGD(lr = Init_lr_fit, momentum = 0.9, nesterov=True)

        model.compile(
            loss={'Embedding' : triplet_loss(batch_size=batch_size//3), 'Softmax' : 'categorical_crossentropy'}, 
            optimizer = optimizer, metrics = {'Softmax' : 'categorical_accuracy'}
        )

        # 获得学习率下降的公式
        lr_scheduler_func = get_lr_scheduler(Init_lr_fit, Min_lr_fit, Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        # 划分训练集和验证集
        train_dataset = Dataset(input_shape, lines[:num_train], batch_size, num_classes, random = True)
        val_dataset = Dataset(input_shape, lines[num_train:], batch_size, num_classes, random = False)

        # 训练参数的设置
        # checkpoint      用于设置权值保存的细节，period用于修改多少epoch保存一次
        # lr_scheduler       用于设置学习率下降的方式
        # early_stopping  用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛

        # 日期格式转化为字符串格式
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')

        # 将tensorflow程序输出的日志文件的信息可视化
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        logging = TensorBoard(log_dir)

        # loss可视化
        loss_history = LossHistory(log_dir)

        # 每次迭代后保存模型
        checkpoint = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)

        # 避免网络发生过拟合的正则化方法
        early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)

        # 学习率时间表旨在通过根据预定义的时间表降低学习率来调整训练期间的学习率。 常见的学习率时间表包括基于时间的衰减，阶跃衰减和指数衰减。
        lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose = 1)

        # LFW估计
        lfw_callback = LFW_callback(LFWDataset(dir="lfw", pairs_path="model_data/lfw_pair.txt", batch_size=32, input_shape=input_shape))

        # 回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。
        callbacks = [logging, loss_history, checkpoint, lr_scheduler, lfw_callback]

        # epoch:迭代次数
        # steps_per_epoch:每次迭代的步数
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator = train_dataset,
            steps_per_epoch = epoch_step,
            validation_data = val_dataset,
            validation_steps = epoch_step_val,
            epochs = Epoch,
            initial_epoch = 0,
            use_multiprocessing = True,
            workers = 1,
            callbacks = callbacks
        )
