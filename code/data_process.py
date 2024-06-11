import pandas as pd
import numpy as np
import tensorflow as tf
from config import cfg
from scipy import signal
import numpy.fft as nf
# from sklearn.preprocessing import StandardScaler

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value))  # 本身就给得是数组，这里不加[]


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def open_excel(filename):
    """
    打开数据集，进行数据处理
    :param filename:文件名
    :return:特征集数据、标签集数据
    """
    num=filename
    # readbook = pd.read_excel(f'./rawdata/{filename}.xlsx', engine='openpyxl',header = None)
    readbook = pd.read_excel(f'D:\code\sleepposture\sleepposture\data\{filename}.xlsx', engine='openpyxl', header=None)
    # print(readbook.loc[0:920-1].values)
    # print(len(readbook.loc[0:920-1].values))
    # print(readbook.loc[0:920-1].values.shape)
    # readbook = list(readbook)
    # print(readbook)
    # print(type(readbook))
#     .T：转置 to_numpy：转换为 NumPy 数组
    nplist = readbook.T
    data = nplist[0:cfg.column].T
#   data = np.float64(data)
#   print(data[0])
    # print("-----------")
    # data = readbook.to_numpy()
    # data = np.float64(data)
    # print(data)
    target = int(num) % cfg.category
    return data, target


def gennerate_terecord_file(tfrecordtrainfilename,tfrecordvalfilename,start,end):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    batch_size = cfg.one_batch_size
    val_rate = cfg.val_rate
    with tf.io.TFRecordWriter(tfrecordtrainfilename) as trainwriter:
        with tf.io.TFRecordWriter(tfrecordvalfilename) as valwriter:
            for i in range(start, end):
                print(i)
                raw_data, label = open_excel(str(i))
                raw_data_len = len(raw_data)
                # print(raw_data_len)
                batch = int(raw_data_len / batch_size)
                # print(batch)

                modnum = int(batch / (batch * (val_rate)))
                # print("modnum",modnum)

                index = 0
                for j in range(batch):
                    batch_data = raw_data.loc[index:index+batch_size - 1].values
                    # flatten()是对多维数据的降维函数
                    batch_data = np.float32(batch_data.flatten())
                    # batch_data = np.float32(batch_data).to_numpy()
                    # print(type(batch_data))
                    index = index + batch_size - 1
                    # print(index)
                    feature = {
                        'data': _float_feature(batch_data),
                        'label': _int64_feature(label)
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature))
                    if j % modnum != 0:
                        trainwriter.write(example.SerializeToString())
                    else:
                        valwriter.write(example.SerializeToString())
            trainwriter.close()
            valwriter.close()

def   gennerate_terecord_file_energy(tfrecordtrainfilename,tfrecordvalfilename,start,end):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    batch_size = cfg.one_batch_size
    val_rate = cfg.val_rate
    with tf.io.TFRecordWriter(tfrecordtrainfilename) as trainwriter:
        with tf.io.TFRecordWriter(tfrecordvalfilename) as valwriter:
            for i in range(start, end):
                print(i)
                raw_data, label = open_excel(str(i))
                raw_data_len = len(raw_data)
                # print(raw_data_len)
                batch = int(raw_data_len / batch_size)
                # print(batch)

                modnum = int(batch / (batch * (val_rate)))
                # print("modnum",modnum)

                index = 0
                for j in range(batch):
                    b_data = raw_data.loc[index:index + batch_size - 1].values
                    batch_data = []
                    # 求能量
                    for r in range(0, cfg.row):
                        s = 125
                        energy = []
                        for clo in range(0, cfg.column):
                            r_data = b_data[s * r:(r + 1) * s, clo]
                            complex_ary = nf.fft(r_data)
                            fft_pow = np.abs(complex_ary) / s * 2
                            fft_pow[0] = fft_pow[0] / 2
                            fft_pow = fft_pow[0:int(s / 2)]
                            ener = np.sum(fft_pow ** 2)
                            energy.append(ener)
                        batch_data.extend(energy)
                    batch_data = np.array(batch_data)
                    # flatten()是对多维数据的降维函数
                    batch_data = np.float32(batch_data.flatten())
                    # batch_data = np.float32(batch_data).to_numpy()
                    # print(type(batch_data))
                    index = index + batch_size - 1
                    # print(index)
                    feature = {
                        'data': _float_feature(batch_data),
                        'label': _int64_feature(label)
                    }
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature))
                    if j % modnum != 0:
                        trainwriter.write(example.SerializeToString())
                    else:
                        valwriter.write(example.SerializeToString())
            trainwriter.close()
            valwriter.close()

def gennerate_terecord_file_1fold(tfrecordtrainfilename, tfrecordvalfilename,tfrecordtestfilename,start, end, k):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    batch_size = cfg.one_batch_size
    val_rate = cfg.val_rate
    # scaler = StandardScaler()
    with tf.io.TFRecordWriter(tfrecordtrainfilename) as trainwriter:
        with tf.io.TFRecordWriter(tfrecordvalfilename) as valwriter:
            with tf.io.TFRecordWriter(tfrecordtestfilename) as testwriter:
                for i in range(start, end):
                    print(i)
                    raw_data, label = open_excel(str(i))
                    raw_data_len = len(raw_data)
                    # print(raw_data_len)
                    batch = int(raw_data_len / batch_size)
                    # print(batch)

                    modnum = int(batch / (batch * (val_rate)))
                    # print("modnum",modnum)
                    index = 0
                    # 归一化
                    # scaler.fit_transform(np.float32(raw_data.loc[0:112500].values))
                    if i >= k*4 and i < k*4+4:
                        print("test :", i)
                        for j in range(batch):
                            batch_data = raw_data.loc[index:index + batch_size - 1].values
                            # 归一化
                            # batch_data = scaler.transform(batch_data)
                            # flatten()是对多维数据的降维函数
                            batch_data = np.float32(batch_data.flatten())
                            # batch_data = np.float32(batch_data).to_numpy()
                            # print(type(batch_data))
                            index = index + batch_size - 1
                            # print(index)
                            feature = {
                                'data': _float_feature(batch_data),
                                'label': _int64_feature(label)
                            }
                            example = tf.train.Example(
                                features=tf.train.Features(feature=feature))
                            testwriter.write(example.SerializeToString())
                    else:
                        print("train :", i)
                        for j in range(batch):
                            batch_data = raw_data.loc[index:index + batch_size - 1].values
                            # 归一化
                            # batch_data = scaler.transform(batch_data)
                            # flatten()是对多维数据的降维函数
                            batch_data = np.float32(batch_data.flatten())
                            # batch_data = np.float32(batch_data).to_numpy()
                            # print(type(batch_data))
                            index = index + batch_size - 1
                            # print(index)
                            feature = {
                                'data': _float_feature(batch_data),
                                'label': _int64_feature(label)
                            }
                            example = tf.train.Example(
                                features=tf.train.Features(feature=feature))
                            if j % modnum != 0:
                                trainwriter.write(example.SerializeToString())
                            else:
                                valwriter.write(example.SerializeToString())
                trainwriter.close()
                valwriter.close()
                testwriter.close()

def condagennerate_terecord_file_1fold_energy(tfrecordtrainfilename, tfrecordvalfilename,tfrecordtestfilename,start, end, k):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    batch_size = cfg.one_batch_size
    val_rate = cfg.val_rate
    # scaler = StandardScaler()
    with tf.io.TFRecordWriter(tfrecordtrainfilename) as trainwriter:
        with tf.io.TFRecordWriter(tfrecordvalfilename) as valwriter:
            with tf.io.TFRecordWriter(tfrecordtestfilename) as testwriter:
                for i in range(start, end):
                    print(i)
                    raw_data, label = open_excel(str(i))
                    raw_data_len = len(raw_data)
                    # print(raw_data_len)
                    batch = int(raw_data_len / batch_size)
                    # print(batch)

                    modnum = int(batch / (batch * (val_rate)))
                    # print("modnum",modnum)
                    index = 0
                    # 归一化
                    # scaler.fit_transform(np.float32(raw_data.loc[0:112500].values))
                    if i >= k*4 and i < k*4+4:
                        print("test :", i)
                        for j in range(batch):
                            b_data = raw_data.loc[index:index + batch_size - 1].values
                            batch_data = []
                            # 求能量
                            for r in range(0, cfg.row):
                                s = 125
                                energy = []
                                for clo in range(0, cfg.column):
                                    r_data = b_data[s*r:(r+1)*s, clo]
                                    complex_ary = nf.fft(r_data)
                                    fft_pow = np.abs(complex_ary) / s * 2
                                    fft_pow[0] = fft_pow[0] / 2
                                    fft_pow = fft_pow[0:int(s/2)]
                                    ener = np.sum(fft_pow ** 2)
                                    energy.append(ener)
                                batch_data.extend(energy)
                            batch_data = np.array(batch_data)
                            # print(batch_data)
                            # flatten()是对多维数据的降维函数
                            batch_data = np.float32(batch_data.flatten())
                            # batch_data = np.float32(batch_data).to_numpy()
                            # print(type(batch_data))
                            index = index + batch_size - 1
                            # print(index)
                            feature = {
                                'data': _float_feature(batch_data),
                                'label': _int64_feature(label)
                            }
                            example = tf.train.Example(
                                features=tf.train.Features(feature=feature))
                            testwriter.write(example.SerializeToString())
                    else:
                        print("train :", i)
                        for j in range(batch):
                            b_data = raw_data.loc[index:index + batch_size - 1].values
                            batch_data = []
                            # 求能量
                            for r in range(0, cfg.row):
                                s = 125
                                energy = []
                                for clo in range(0, cfg.column):
                                    r_data = b_data[s*r:(r+1)*s, clo]
                                    complex_ary = nf.fft(r_data)
                                    fft_pow = np.abs(complex_ary) / s * 2
                                    fft_pow[0] = fft_pow[0] / 2
                                    fft_pow = fft_pow[0:int(s/2)]
                                    ener = np.sum(fft_pow ** 2)
                                    energy.append(ener)
                                batch_data.extend(energy)
                            batch_data = np.array(batch_data)
                            # print(batch_data)
                            # flatten()是对多维数据的降维函数
                            batch_data = np.float32(batch_data.flatten())
                            # batch_data = np.float32(batch_data).to_numpy()
                            # print(type(batch_data))
                            index = index + batch_size - 1
                            # print(index)
                            feature = {
                                'data': _float_feature(batch_data),
                                'label': _int64_feature(label)
                            }
                            example = tf.train.Example(
                                features=tf.train.Features(feature=feature))
                            if j % modnum != 0:
                                trainwriter.write(example.SerializeToString())
                            else:
                                valwriter.write(example.SerializeToString())
                trainwriter.close()
                valwriter.close()
                testwriter.close()


def gennerate_terecord_file_1fold_filter(tfrecordtrainfilename, tfrecordvalfilename,tfrecordtestfilename,start, end, k):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    batch_size = cfg.one_batch_size
    val_rate = cfg.val_rate
    # scaler = StandardScaler()
    with tf.io.TFRecordWriter(tfrecordtrainfilename) as trainwriter:
        with tf.io.TFRecordWriter(tfrecordvalfilename) as valwriter:
            with tf.io.TFRecordWriter(tfrecordtestfilename) as testwriter:
                for i in range(start, end):
                    print(i)
                    raw_data, label = open_excel(str(i))
                    raw_data_len = len(raw_data)
                    # print(raw_data_len)
                    batch = int(raw_data_len / batch_size)
                    # print(batch)

                    modnum = int(batch / (batch * (val_rate)))
                    # print("modnum",modnum)
                    index = 0
                    # 归一化
                    # scaler.fit_transform(np.float32(raw_data.loc[0:112500].values))
                    if i >= k*4 and i < k*4+4:
                        print("test :", i)
                        for j in range(batch):
                            batch_data = raw_data.loc[index:index + batch_size - 1].values
                            b, a = signal.butter(1, [0.001, 0.1], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
                            after_filter_data = []
                            for i in range(cfg.column):
                                data = np.float32(batch_data).T
                                after_filter_data.extend(signal.filtfilt(b, a, data[i]))  # data为要过滤的信号
                            batch_data = np.array(after_filter_data)
                            # 归一化
                            # batch_data = scaler.transform(batch_data)
                            # flatten()是对多维数据的降维函数
                            batch_data = np.float32(batch_data.T.flatten())
                            # batch_data = np.float32(batch_data).to_numpy()
                            # print(type(batch_data))
                            index = index + batch_size - 1
                            # print(index)
                            feature = {
                                'data': _float_feature(batch_data),
                                'label': _int64_feature(label)
                            }
                            example = tf.train.Example(
                                features=tf.train.Features(feature=feature))
                            testwriter.write(example.SerializeToString())
                    else:
                        print("train :", i)
                        for j in range(batch):
                            batch_data = raw_data.loc[index:index + batch_size - 1].values

                            b, a = signal.butter(1, [0.001, 0.1], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
                            after_filter_data = []
                            for i in range(cfg.column):
                                data = np.float32(batch_data).T
                                after_filter_data.extend(signal.filtfilt(b, a, data[i]))  # data为要过滤的信号
                            batch_data = np.array(after_filter_data)
                            # 归一化
                            # batch_data = scaler.transform(batch_data)
                            # flatten()是对多维数据的降维函数
                            batch_data = np.float32(batch_data.T.flatten())
                            # batch_data = np.float32(batch_data).to_numpy()
                            # print(type(batch_data))
                            index = index + batch_size - 1
                            # print(index)
                            feature = {
                                'data': _float_feature(batch_data),
                                'label': _int64_feature(label)
                            }
                            example = tf.train.Example(
                                features=tf.train.Features(feature=feature))
                            if j % modnum != 0:
                                trainwriter.write(example.SerializeToString())
                            else:
                                valwriter.write(example.SerializeToString())
                trainwriter.close()
                valwriter.close()
                testwriter.close()


def gennerate_terecord_test_file(tfrecordvalfilename,start,end):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    batch_size = cfg.one_batch_size
    with tf.io.TFRecordWriter(tfrecordvalfilename) as valwriter:
        for i in range(start, end):
            print(i)
            raw_data, label = open_excel(str(i))
            raw_data_len = len(raw_data)
            # print(raw_data_len)
            batch = int(raw_data_len / batch_size)
            # print(batch)

            index = 0
            for j in range(batch):
                batch_data = raw_data.loc[index:index+batch_size - 1].values
                # flatten()是对多维数据的降维函数
                batch_data = np.float32(batch_data.flatten())
                # batch_data = np.float32(batch_data).to_numpy()
                # print(type(batch_data))
                index = index + batch_size - 1
                # print(index)
                feature = {
                    'data': _float_feature(batch_data),
                    'label': _int64_feature(label)
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                valwriter.write(example.SerializeToString())
        valwriter.close()

def gennerate_terecord_file_for5Kfold(tfrecordtrainfilename,tfrecordvalfilename,start,end,valk):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    index_k = 1
    one_batch_size = cfg.one_batch_size
    with tf.io.TFRecordWriter(tfrecordtrainfilename) as trainwriter:
        with tf.io.TFRecordWriter(tfrecordvalfilename) as valwriter:
            for i in range(start, end):
                print(i)
                raw_data, label = open_excel(str(i))
                raw_data_len = len(raw_data)
                batch = int(raw_data_len / one_batch_size)

            index = 0
            for j in range(batch):
                batch_data = raw_data.loc[index:index + one_batch_size - 1].values
                # flatten()是对多维数据的降维函数
                batch_data = np.float32(batch_data.flatten())
                # batch_data = np.float32(batch_data).to_numpy()
                # print(type(batch_data))
                index = index + one_batch_size - 1
                # print(index)
                feature = {
                    'data': _float_feature(batch_data),
                    'label': _int64_feature(label)
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                if index_k == valk:
                    valwriter.write(example.SerializeToString())
                else:
                    trainwriter.write(example.SerializeToString())
                if index_k >= 6:
                    index_k = 0
                index_k = index_k + 1
            trainwriter.close()
            valwriter.close()


# tfrecord文件的映射函数
def map_func(example):
    # feature 的属性解析表
    # 数据和标签保存和读取要一致
    feature_map = {
                   'data': tf.io.FixedLenFeature([cfg.row*cfg.column], tf.float32),
                   'label': tf.io.FixedLenFeature([1], tf.int64)
                   }
    parsed_example = tf.io.parse_single_example(example, features=feature_map)
    data = parsed_example['data']
    # data = tf.io.decode_raw(parsed_example['data'], out_type=tf.float32)
    label = parsed_example['label']
    return data, label


def _parse_example(example_string):
    """
    tfrecod数据解析
    :param example_string:
    :return:
    """
    # 浮点型数组时不能用FixedLenFeature，要用FixedLenSequenceFeature！！！！！！！
    feature_description = {
        'data': tf.io.FixedLenFeature([cfg.row*cfg.column], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    feature_dict = tf.io.parse_example(example_string,
                                       feature_description)
    # 实际上不用归一化也一样
    # scaler = StandardScaler()
    # print(feature_dict['data'])
    # data = scaler.fit_transform(feature_dict['data'].astype(np.float32).reshape(-1, 1)).reshape(cfg.row, cfg.column, 1)
    data = tf.reshape(feature_dict['data'], [cfg.row, cfg.column, 1])
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    labels = feature_dict['label']
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    # labels = tf.cast(labels, tf.int64)

    return data, labels


def gen_data_batch(file_pattern, batch_size, is_training=True):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        # dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        dataset = dataset.repeat()
        dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def gen_valdata_batch(file_pattern, batch_size):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.repeat(1)
    dataset = dataset.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_truelabel(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(map_func=map_func)
    truelabel = []
    for data,label in dataset:
        label = np.int64(label)
        truelabel.extend(label)
    return truelabel

if __name__ == '__main__':
    open_excel(str(0))
    # tfrecord_train = cfg.train_dataset_path
    # tfrecord_val = cfg.val_dataset_path
    # # gennerate_terecord_file_for5Kfold(0,21)
    # gennerate_terecord_file(tfrecord_train,tfrecord_val,  0, 56)
    # gennerate_terecord_test_file(cfg.test_dataset_path, 56, 60)
    # dataset1 = tf.data.TFRecordDataset(tfrecord_val)
    # dataset1 = dataset1.map(map_func=map_func)
    # print(dataset1)
    #
    # train_data = []
    # val_data = []
    # i = 0
    # for data, label in dataset1:
    #     i = i+1
    #     print(i)
    #     print(data)
    #     print(label)
    # train_dataset = gen_data_batch(cfg.train_dataset_path,
    #                                cfg.batch_size,
    #                                is_training=True)
    # print(train_dataset)



