#coding:utf-8

import numpy as np
import scipy.io as sio
import scipy.misc as misc
import sys
import os
import scipy.ndimage as ndimage
import scipy.signal as signal
import sklearn.preprocessing as preproc
import struct

def load_coil20():
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


def load_mnist():
	'''
		load minist img
	'''
	labels_path = sys.path[0] + '/../dataset/train-labels.idx1-ubyte'
	images_path = sys.path[0] + '/../dataset/train-images.idx3-ubyte'
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II',lbpath.read(8))
		labels = np.fromfile(lbpath,dtype=np.uint8)
	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
		images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
	sio.savemat(sys.path[0] + '/../dataset/minist.mat', {'img': images, 'gnd' : labels})


load_mnist()