#coding:utf-8

from __future__ import division, absolute_import
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np



def gen_conv_encoder(encoder_para):
    '''
        generate a one layer convolutional encoder

        @encoder_para: dict, parameters of method, including:
            @@img_size: np.ndarray, shape=[2:], the size of input image
            @@kernal_size: np.ndarray, shape=[4:], [width, height, input_channel, output_channel]
            @@strides: np.ndarray, shape=[4:], [batch_stride, width_stride, height_stride, channel_stride]
    '''

    img_size = encoder_para['img_size']
    kernal_size = encoder_para['kernal_size']
    strides = encoder_para['strides']

    # generate X and weight
    X = tf.placeholder(tf.float32, [None, img_size[0], img_size[1], 1]) #batch_size, img_width, img_height, channel
    enc_knl = tf.get_variable("enc_knl", shape=kernal_size, \
    initializer=layers.xavier_initializer_conv2d())
    enc_b = tf.Variable(tf.zeros([kernal_size[-1]], dtype=tf.float32))
    #generate conv layer
    conv_layer = tf.nn.conv2d(X, enc_knl, strides, padding='SAME')
    conv_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, enc_b))
    return conv_layer, X

def gen_conv_decoder(X, decoder_para):
    '''
        generate a one layer deconvolutional decoder

        @X: input data
        @decoder_para: dict, parameters of method, including:
            @@img_size: list, shape=[4:], the size of input image
            @@kernal_size: list, shape=[4:], [width, height, input_channel, output_channel]
            @@strides: list, shape=[4:], [batch_stride, width_stride, height_stride, channel_stride]
            @@output_size: list, shape=[4:], [batch, width, height, channel]
    '''

    # generate X and weight
    img_size = decoder_para['img_size']
    kernal_size = decoder_para['kernal_size']
    strides = decoder_para['strides']
    output_size = decoder_para['output_size']


    dec_knl = tf.get_variable("dec_knl", shape=kernal_size, \
    initializer=layers.xavier_initializer_conv2d())
    dec_b = tf.Variable(tf.zeros([1], dtype=tf.float32))
    # generate deconvolutional layer
    deconv_layer = tf.nn.conv2d_transpose(X, dec_knl, tf.stack(output_size), strides=strides, padding='SAME')
    deconv_layer = tf.nn.relu(tf.add(deconv_layer, dec_b))
    return deconv_layer




    


