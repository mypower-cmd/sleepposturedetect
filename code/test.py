import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.fft as nf
# from data_process import *
#
#
# raw_data, label = open_excel(str(1))
# batch_data = raw_data.loc[0:128 - 1].values
# # print(batch_data[:,1])
# complex_ary = nf.fft(batch_data[:,4])
# # print(complex_ary)
# fft_pow = np.abs(complex_ary) / 128 * 2
# fft_pow[0] = fft_pow[0] / 2
# # print(fft_pow)
# list1 = np.array(range(0, 64))
# freq1 = 128*list1/64        # 单边谱的频率轴
#
# # 绘制结果
# plt.figure()
# plt.plot(freq1, fft_pow[0:64])
# plt.title('fft')
# plt.xlabel('frequency  (Hz)')
# plt.ylabel(' Amplitude ')
# plt.show()
# # fft_pow = fft_pow[0:64]
# energy = np.sum(fft_pow**2)
# print(energy)
# from tensorflow.keras import layers
# # from keras.layers import (Input, Reshape)
# input = layers.Input(shape=(6, 6, 1))
# x = layers.Conv2D(32, (3, 3))(input)
# x = layers.MaxPooling2D()(x)
# x = layers.Conv2D(64, (1, 1))(x)
# x = layers.MaxPooling2D()(x)
# print(x)