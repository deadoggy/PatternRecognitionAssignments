#coding:utf-8

import numpy as np
import scipy.io as sio
import scipy.misc as misc
import sys
import os
import scipy.ndimage as ndimage
import scipy.signal as signal
import sklearn.preprocessing as preproc
#read all img file in dataset folder

walker = os.walk(sys.path[0] + '/../dataset/coil-20-proc/')
img_list = []
gnd_list = []

for dir_path, sub_folders, sub_files in walker:
    for imgf in sub_files:
        gnd = int(imgf[3:imgf.index('__')])
        img = signal.medfilt(misc.imread(dir_path + imgf), kernel_size=[3,3])
        downsample_img = preproc.MinMaxScaler().fit_transform(ndimage.zoom(img, 0.25, mode='mirror'))
        img_list.append(np.reshape(downsample_img, 1024))
        gnd_list.append(gnd)

sio.savemat(sys.path[0] + '/../dataset/coil-20.mat', {'img': img_list, 'gnd' : gnd_list})