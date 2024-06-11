from config import cfg
from tensorflow.keras.callbacks import (ReduceLROnPlateau)
from model import *
import matplotlib.pyplot as plt
from data_process import *
import datetime

#配置GPU，限制GPU内存增长
def GPU_Config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def train_and_val():
    # 训练集和测试集
    train_x = []
    train_y = []
    # 配置GPU
    GPU_Config()

    # 创建模型
    model = cnn_model()

    # Adam优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train_learning_rate)
    # 交叉熵损失函数
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # 配置训练时用的优化器、损失函数和准确率评测标准
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    #打印神经网络结构，统计参数数目
    model.summary()

    train_dataset_list = []

    # 获取数据
    train_dataset1 = gen_data_batch(cfg.train_dataset_path+'1',
                                   cfg.batch_size,
                                   is_training=True)
    train_dataset_list.extend(train_dataset1)
    train_dataset2 = gen_data_batch(cfg.train_dataset_path+'2',
                                   cfg.batch_size,
                                   is_training=True)
    train_dataset_list.extend(train_dataset2)
    train_dataset3 = gen_data_batch(cfg.train_dataset_path+'3',
                                   cfg.batch_size,
                                   is_training=True)
    train_dataset_list.extend(train_dataset3)
    train_dataset4 = gen_data_batch(cfg.train_dataset_path+'4',
                                   cfg.batch_size,
                                   is_training=True)
    train_dataset_list.extend(train_dataset4)
    train_dataset5 = gen_data_batch(cfg.train_dataset_path+'5',
                                   cfg.batch_size,
                                   is_training=True)
    train_dataset_list.extend(train_dataset5)
    print('train_dataset_list len:', len(train_dataset_list))

    # 配置回调函数
    callback = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('./model/posture_classify.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ]

    # historys用来记录5组loss，auc，val_loss，val_auc
    historys = []

    for i in range(5):
        train_dataset = train_dataset1 + train_dataset2 + train_dataset3 + train_dataset4
        val_dataset = train_dataset5
        # index = -1
        # flag = -1
        # for j in range(5):
        #     if j != i:
        #         if index == -1:
        #             index = j
        #         elif index != -1 and flag == -1:
        #             x, y = train_dataset_list[index]
        #             x2, y2 = train_dataset_list[j]
        #             train_x = tf.concat([x, x2], 0)
        #             train_y = tf.concat([y, y2], 0)
        #             # print(tf.shape(train_dataset))
        #             flag = 1
        #         else:
        #             x2, y2 = train_dataset_list[j]
        #             train_x = tf.concat([train_x, x2], 0)
        #             train_y = tf.concat([train_y, y2], 0)
        #
        # # train_x, train_y = train_dataset_list[0]
        # # print(train_dataset_list)
        # val_x, val_y = train_dataset_list[i]
        # print("--------------shape----------------")
        # print(tf.shape(train_x))
        # print(tf.shape(train_y))
        # print(tf.shape(val_x))
        # print(tf.shape(val_y))
        # print(len(train_x))
        # print("------------------------------")
        # 模型训练
        history = model.fit(train_dataset,
                            epochs=cfg.epochs,
                            # validation_split=0.125,
                            # validation_data=[x_test, y_test],
                            validation_data=val_dataset,
                            callbacks=callback,
                            shuffle=True,
                            verbose=1,
                            validation_freq=1
                            # steps_per_epoch=cfg.train_num_samples / cfg.batch_size,
                            # validation_steps=cfg.val_num_samples / cfg.batch_size
                            )
        historys.append(history)

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.plot(range(cfg.epochs), historys[i].history['accuracy'], label='Training Accuracy')
        plt.plot(range(cfg.epochs), historys[i].history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
    plt.title("Training and Validation Accuracy")
    result_png = "./result_img/" + "result_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.plot(range(cfg.epochs), historys[i].history['loss'], label='Training Loss')
        plt.plot(range(cfg.epochs), historys[i].history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # 保存图片
    result_png = "./result_img/" + "result_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()







if __name__ == '__main__':
    train_and_val()


