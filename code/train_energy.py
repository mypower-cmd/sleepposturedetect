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
            print(len(gpus), "Physical GPUs,", len(logical_gpus),"Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def train_and_val():
    # 读取验证训练集和测试集长度
    list_num_samples = np.load(cfg.train_num_samples_file_name)  # 读取
    cfg.train_num_samples = list_num_samples[0]
    cfg.val_num_samples = list_num_samples[1]
    print("train_num_samples:", cfg.train_num_samples)
    print("val_num_samples：", cfg.val_num_samples)
    train_x = []
    train_y = []
    # 配置GPU
    # GPU_Config()

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

    # 获取数据
    train_dataset = gen_data_batch(cfg.train_dataset_path,
                                   cfg.batch_size,
                                   is_training=True)
    # val_dataset = gen_data_batch(cfg.val_dataset_path,
    #                              cfg.batch_size,
    #                              is_training=False)

    val_data = train_dataset.skip((cfg.train_num_samples*(1-cfg.val_rate))//cfg.batch_size)
    train_data = train_dataset.take((cfg.train_num_samples*(1-cfg.val_rate))//cfg.batch_size)

    # 配置回调函数
    callback = [
        # ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('../model/posture_classify.h5', monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ]


    # 模型训练
    history = model.fit(train_data,
                        epochs=cfg.epochs,
                        # validation_split=0.2,
                        batch_size=cfg.batch_size,
                        callbacks=callback,
                        shuffle=True,
                        verbose=1,
                        validation_data=val_data,
                        validation_freq=1,
                        # steps_per_epoch=int(cfg.train_num_samples / cfg.batch_size),
                        # validation_steps=int(cfg.val_num_samples / cfg.batch_size)
                        steps_per_epoch = ((cfg.train_num_samples * (1 - cfg.val_rate)) // cfg.batch_size),
                         validation_steps = ((cfg.train_num_samples * cfg.val_rate) // cfg.batch_size),
                        )

    pd.DataFrame(history.history).to_csv('training_log.csv', index=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 画出训练和测试的准确率和损失值
    epochs_range = range(cfg.epochs)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # 保存图片12
    result_png = "../result_img/" + "result_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()

    test_dataset = gen_data_batch(cfg.val_dataset_path,
                                  cfg.batch_size,
                                  is_training=False)


    print('-----------test--------')
    test_loss, test_acc = model.evaluate(test_dataset, steps=(cfg.val_num_samples // cfg.batch_size), verbose=1)
    print('测试准确率：', test_acc)
    print('测试损失', test_loss)
    # 模型保存
    # model_path = "./model/posture_classify.h5"
    # model.save(model_path)







if __name__ == '__main__':
    # GPU_Config()
    # gennerate_terecord_trainfile(cfg.train_dataset_path, cfg.val_dataset_path)
    train_and_val()
    # model = cnn_model()
    # 打印神经网络结构，统计参数数目
    # model.summary()


