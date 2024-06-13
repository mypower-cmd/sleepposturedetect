import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.fft as nf
from data_process import *
import os

if __name__ == '__main__':
    filenames = os.listdir(r"../raw_data/")
    print(filenames)
    print(type(filenames))
    for filename in filenames:
        print(filename)
        if filename.endswith('m.xlsx'):
            print('0')
        elif filename.endswith('left.xlsx'):
            print('1')
        elif filename.endswith('right.xlsx'):
            print('2')
        # elif filename.endswith('motion.xlsx'):
        #     print('3')
        else:
            continue
        data = read_xlsx(f'../raw_data/{filename}')
        print(data)
        print(type(data))
        print(np.shape(data))


