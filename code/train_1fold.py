from config import cfg
from tensorflow.keras.callbacks import (ReduceLROnPlateau)
from model import *
import matplotlib.pyplot as plt
from data_process import *
import datetime
import os

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

historys = []
test_accs = []
test_losses = []
time_str = ''
model_path = ''
def train_and_val():
    # 配置GPU
    # GPU_Config()
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("load_model")
    else:
        # 创建模型
        print("create_model")
        model = cnn_model()

        # Adam优化器
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.train_learning_rate)
        # 交叉熵损失函数
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # 配置训练时用的优化器、损失函数和准确率评测标准
        # model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])
        model.compile(optimizer=optimizer, loss=loss, metrics=[
            'accuracy'
            # tf.keras.metrics.Accuracy()
            # tf.keras.metrics.SparseCategoricalCrossentropy()
        ])

    #打印神经网络结构，统计参数数目
    model.summary()

    # 获取数据
    train_dataset = gen_data_batch(cfg.train_dataset_path,
                                   cfg.batch_size,
                                   is_training=True)
    val_dataset = gen_data_batch(cfg.val_dataset_path,
                                 cfg.batch_size,
                                 is_training=False)
    # 配置回调函数
    callback = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ]

    print(train_dataset)
    # 模型训练
    history = model.fit(train_dataset,
                        # batch_size=3,
                        epochs=cfg.epochs,
                        # validation_split=0.125,
                        # validation_data=[x_test, y_test],
                        validation_data=val_dataset,
                        callbacks=callback,
                        shuffle=True,
                        verbose=1,
                        validation_freq=1,
                        steps_per_epoch=int(cfg.train_num_samples // (cfg.one_batch_size)),
                        validation_steps=int(cfg.val_num_samples // (cfg.one_batch_size)))

    historys.append(history)


    pd.DataFrame(history.history).to_csv('training_log.csv', index=True)

    # acc = history.history['sparse_categorical_accuracy']
    # val_acc = history.history['val_sparse_categorical_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

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
    result_png = "./result_img/" + "result_" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()

    test_dataset = gen_data_batch(cfg.test_dataset_path,
                                  cfg.batch_size,
                                  is_training=False)


    print('-----------test--------')
    test_loss, test_acc = model.evaluate(test_dataset, steps=(cfg.test_num_samples // (cfg.one_batch_size)), verbose=1)
    print('测试准确率：', test_acc)
    print('测试损失', test_loss)
    test_accs.append(test_acc)
    test_losses.append(test_loss)
    # 模型保存
    # model_path = "./model/posture_classify.h5"
    # model.save(model_path)







if __name__ == '__main__':
    GPU_Config()
    time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
    model_path = './model/posture_classify'+time_str+'.h5'
    for i in range(0, cfg.people_num):
        gennerate_terecord_file_1fold_filter(cfg.train_dataset_path, cfg.val_dataset_path, cfg.test_dataset_path, 0, cfg.simple_num_file,i)
        train_and_val()

    for i in range(0, cfg.people_num):
        print('使用第n个测试：', i)
        print('测试准确率：', test_accs[i])
        print('测试损失', test_losses[i])

    test_acc = sum(test_accs) / cfg.people_num
    test_loss = sum(test_losses) / cfg.people_num
    print('平均测试准确率：', test_acc)
    print('平均测试损失', test_loss)


    # 画出训练和测试的准确率和损失值
    epochs_range = range(cfg.people_num)
    plt.figure()
    plt.plot(epochs_range, test_accs, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Test Accuracy')
    # 保存图片
    result_png = "./result_img/" + "result_Test_Accuracy" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()


    plt.figure()
    plt.plot(epochs_range, test_losses, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Test Loss')
    # 保存图片12
    result_png = "./result_img/" + "result_Test_Loss" + datetime.datetime.now(
    ).strftime("%m%d_%H%M%S") + ".png"
    plt.savefig(result_png)
    plt.show()


