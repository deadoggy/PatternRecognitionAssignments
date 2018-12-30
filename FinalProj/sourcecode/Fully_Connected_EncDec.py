#coding:utf-8

from __future__ import division, absolute_import
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


def gen_fully_connected_encoder(encoder_para):
    '''
        generate a fully connectde encoder

        @encoder_para: dict, parameters of method, including:
            @@img_size: np.ndarray, shape=[4:], the size of input image
            @@kernal_size: np.ndarray, shape=[4:], [width, height, input_channel, output_channel]
            @@strides: np.ndarray, shape=[4:], [batch_stride, width_stride, height_stride, channel_stride]
            @@fullyconn_outsize: int, out size of fully connected layer
            @batch_size: int, batch size
    '''
    img_size = encoder_para['img_size']
    kernal_size = encoder_para['kernal_size']
    strides = encoder_para['strides']
    fullyconn_outsize = encoder_para['fullyconn_outsize']
    batch_size = encoder_para['batch_size']

    X = tf.placeholder(tf.float32, shape=img_size)
    enc_knl = tf.get_variable('enc_knl', shape=kernal_size, initializer=layers.xavier_initializer_conv2d())
    enc_b = tf.Variable(tf.zeros([kernal_size[-1]], dtype=tf.float32))
    #generate conv layer
    conv_layer = tf.nn.conv2d(X, enc_knl, strides, padding='SAME')
    conv_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, enc_b))
    #generate fully connected layer
    shape_conv = conv_layer.get_shape().as_list()
    expansion_conv_layer = tf.reshape(conv_layer, shape=[batch_size, -1])
    fully_weight = tf.get_variable('fully_weight', \
        shape=[shape_conv[1] * shape_conv[2] * shape_conv[3], fullyconn_outsize],\
        initializer=layers.xavier_initializer())
    fully_b = tf.Variable(tf.zeros([fullyconn_outsize], dtype=tf.float32))
    fully_layer = tf.nn.relu(tf.nn.bias_add(tf.matmul(expansion_conv_layer, fully_weight), fully_b))
    return fully_layer, X


def gen_fully_connected_decoder(X, decoder_para):
    '''
        generate a one layer deconvolutional decoder

        @X: input data
        @decoder_para: dict, parameters of method, including:
            @@img_size: np.ndarray, shape=[2:], the size of input image
            @@kernal_size: list, shape=[4:], [width, height, input_channel, output_channel]
            @@strides: list, shape=[4:], [batch_stride, width_stride, height_stride, channel_stride]
            @@output_size: list, shape=[4:], [batch, width, height, channel]
            @@fullyconn_insize: int, input size of fully connected layer
    '''
    img_size = decoder_para['img_size']
    kernal_size = decoder_para['kernal_size']
    strides = decoder_para['strides']
    output_size = decoder_para['output_size']
    fullyconn_insize = decoder_para['fullyconn_insize']

    # shape of X is [batch_size, fullyconn_insize]
    # the out size of fully connected layer is (img_size[1]/width_stride) * (img_size[2]/height_stride) * output_channel
    fullyconn_outsize = (img_size[1]/strides[1]) * (img_size[2]/strides[2]) * kernal_size[3] 
    fully_weight = tf.get_variable('dec_fully_weight', \
        shape=[fullyconn_insize, fullyconn_outsize],\
        initializer=layers.xavier_initializer())
    fully_b = fully_b = tf.Variable(tf.zeros([fullyconn_outsize], dtype=tf.float32))
    fully_layer = tf.nn.relu(tf.nn.bias_add(
        tf.matmul(X, fully_weight), fully_b
    ))

    #reshape fully_layer to deconv
    deconv_input = tf.reshape(fully_layer, shape=\
        [-1, int(img_size[1]/strides[1]), int(img_size[2]/strides[2]), kernal_size[3]])
    #deconv
    dec_knl = tf.get_variable("dec_knl", shape=kernal_size, \
        initializer=layers.xavier_initializer_conv2d())
    dec_b = tf.Variable(tf.zeros([kernal_size[-1]], dtype=tf.float32))
    # generate deconvolutional layer
    deconv_layer = tf.nn.conv2d_transpose(deconv_input, dec_knl, tf.stack(output_size), strides=strides, padding='SAME')
    deconv_layer = tf.nn.relu(tf.add(deconv_layer, dec_b))
    return deconv_layer

    