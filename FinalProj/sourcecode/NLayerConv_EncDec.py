#coding:utf-8

from __future__ import division, absolute_import
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np



def gen_nlayer_conv_encoder(encoder):
    '''
        generate a two layer convolutional encoder

         @encoder_para: dict, parameters of method, including:
            @@img_size: np.ndarray, shape=[4:], the size of input image
            @@kernal_size: np.ndarray, shape=[4:], [width, height, input_channel, output_channel]
            @@strides: np.ndarray, shape=[4:], [batch_stride, width_stride, height_stride, channel_stride]
    '''


