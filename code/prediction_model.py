import pandas as pd
import numpy as np
import tensorflow as tf
import random

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
    readbook = pd.read_excel(f'./rawdata/{filename}.xlsx', engine='openpyxl',header = None)
    # print(readbook.loc[0:920-1].values)
    # print(len(readbook.loc[0:920-1].values))
    # print(readbook.loc[0:920-1].values.shape)
    # readbook = list(readbook)
    # print(readbook)
    # print(type(readbook))
#     .T：转置 to_numpy：转换为 NumPy 数组
#     nplist = readbook.T.to_numpy()
#     data = nplist[0:8].T
#     data = np.float64(data)
    # print(data)
    # print("-----------")
    # data = readbook.to_numpy()
    # data = np.float64(data)
    # print(data)
    target = int(num)%3
    return readbook, target


def gennerate_terecord_file(tfrecordfilename,start,end):
    """
    生成tfrecord文件
    :param tfrecordfilename: 生成的tfrecord文件名字
    :return:
    """
    batch_size = 920
    with tf.io.TFRecordWriter(tfrecordfilename) as writer:
        for i in range(start, end):
            print(i)
            raw_data, label = open_excel(str(i))
            raw_data_len = len(raw_data)
            # print(raw_data_len)
            batch = int(raw_data_len / batch_size)
            # print(batch)

            index = 0
            for i in range(batch):
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
                writer.write(example.SerializeToString())
        writer.close()

# tfrecord文件的映射函数
def map_func(example):
    # feature 的属性解析表
    # 数据和标签保存和读取要一致
    feature_map = {
                   'data': tf.io.FixedLenFeature([920*8], tf.float32),
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
        'data': tf.io.FixedLenFeature([920 * 8], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }
    feature_dict = tf.io.parse_example(example_string,
                                       feature_description)
    # 实际上不用归一化也一样
    data = tf.reshape(feature_dict['data'], [920, 8, 1])

    labels = feature_dict['label']
    labels = tf.cast(labels, tf.int64)

    return data, labels


def gen_data_batch(file_pattern, batch_size, num_repeat=1, is_training=True):
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    if is_training:
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.map(_parse_example, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(buffer_size=16 * batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.map(_parse_example, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# -----------------------------------------------------
def read_dataset():
    feature=[]
    label=[]
    split_size=920
    for i in range(0, 18):
        f_temp, l_temp = open_excel(str(i))
        print(type(l_temp))
        # data_list,target_list=split_data(f_temp, l_temp, split_size)
        # feature.extend(data_list)
        # label.extend(target_list)
    return feature, label


def split_data(data,target,split_size):
    data_list=[]
    target_list=[]
    print(len(data))
    x=int((112520-320)/(split_size-320))
    for i in range(x):
        data_list.append(data[i*(split_size-320):i*(split_size-320)+split_size])
        target_list.append(target)
    return data_list,target_list

def random_number(data_size, key):
    """
   使用shuffle()打乱
    """
    number_set = []
    for i in range(data_size):
        number_set.append(i)

    if key == 1:
        random.shuffle(number_set)

    return number_set


def split_data_set(data_set, target_set,rate, ifsuf):
    """
    说明：分割数据集，默认数据集的rate是测试集
    :param data_set: 数据集
    :param target_set: 标签集
    :param rate: 测试集所占的比率
    :return: 返回训练集数据、测试集数据、训练集标签、测试集标签
    """
    # 计算训练集的数据个数
    train_size = int(len(data_set)*(1-rate))
    # 随机获得数据的下标
    data_index = random_number(len(data_set), ifsuf)
    # 分割数据集（X表示数据，y表示标签），以返回的index为下标
    # 训练集数据
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    for i in range(0,train_size):
        x_train.append(data_set[data_index[i]])
        # 测试集数据
        y_train.append(target_set[data_index[i]])
        # 训练集标签
    for i in range(train_size,len(data_index)):
        x_test.append(data_set[i])
        # 测试集标签
        y_test.append(target_set[i])
    print("end")

    return x_train, x_test, y_train, y_test
if __name__ == '__main__':
    tfrecord_train = "train.tfrecords"
    tfrecor_val = "val.tfrecords"
    # gennerate_terecord_file(tfrecord_train, 0, 12)
    # gennerate_terecord_file(tfrecor_val, 12, 18)
    # train_dataset, val_dataset = gen_data_batch(tfrecord_train,
    #                                10,
    #                                10,
    #                                is_training=True)
    # print(train_dataset)

    dataset = tf.data.TFRecordDataset(tfrecord_train)
    dataset = dataset.map(map_func=map_func)
    print(dataset)

    train_data = []
    val_data = []
    i = 0
    for data, label in dataset:
        i = i+1
        print(i)
        print(data)
        print(label)



