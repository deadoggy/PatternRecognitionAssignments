#coding:utf-8

from __future__ import division, absolute_import
import tensorflow as tf
import numpy as np
from sklearn import cluster
import scipy.io as sio
import sys
from Conv_EncDec import gen_conv_decoder, gen_conv_encoder
from DSC_Net import DSC_Net

data_dict = sio.loadmat('/home/yinjia/Documents/Deep-subspace-clustering-networks/Data/COIL20.mat')
imgs = np.reshape(data_dict['fea'], [len(data_dict['fea']), 32, 32, 1])

# data_dict = sio.loadmat(sys.path[0] + '/../dataset/coil-20.mat')
# imgs = np.reshape(data_dict['img'], [len(data_dict['img']), 32, 32, 1])
ground_truth = data_dict['gnd']


def ConvEncDec():
    img_size = [None, 32, 32, 1]
    kernal_size = [3, 3, 1, 15]
    strides = [1, 2, 2, 1]
    reg1 = 1.0
    reg2 = 150.0
    lr = 1e-3

    # encoder, enc_pl = gen_conv_encoder(img_size, kernal_size, stride)
    # decoder, dec_pl = gen_conv_decoder(img_size, kernal_size, stride, [1440, 32, 32, 1])
    encoder = gen_conv_encoder
    decoder = gen_conv_decoder
    encoder_para = {'img_size':img_size, 'kernal_size':kernal_size, 'strides':strides}
    decoder_para = {'img_size':img_size, 'kernal_size':kernal_size, 'strides':strides, 'output_size':[1440, 32, 32, 1]}

    dscnet = DSC_Net(encoder, encoder_para, decoder, decoder_para, len(imgs), img_size, reg1, reg2)
    for itr in xrange(30):
        print itr
        weight_mat, recover_loss, weight_loss, selfexp_loss = dscnet.train(imgs, lr)
        print "l1 %f"%(weight_loss/reg1)
        print "l2 %f"%(selfexp_loss/reg2)
        print "recon_loss %f"%recover_loss
    print 'done'


ConvEncDec()