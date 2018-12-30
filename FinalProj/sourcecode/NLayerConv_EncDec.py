#coding:utf-8

from __future__ import division, absolute_import
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np



def gen_nlayer_conv_encoder(encoder_para):
	'''
		generate a n-layer convolutional encoder

        @encoder_para: dict, parameters of method, including:
            @@img_size: np.ndarray, shape=[4:], the size of input image
            @@kernal_size: np.ndarray, shape=[n:4], [[height, width, input_channel, output_channel], ... ]
            @@strides: np.ndarray, shape=[n:4], [[batch_stride, height_stride, width_stride, channel_stride], ... ]
	'''
	img_size = encoder_para['img_size']
	kernal_size = encoder_para['kernal_size']
	strides = encoder_para['strides']

	n_layers = len(kernal_size)
	X = tf.placeholder(tf.float32, shape=img_size)
	last_layer = X
	for l_idx in xrange(n_layers):
		stride = strides[l_idx]
		knl_w = tf.get_variable('enc_w_%d'%l_idx, shape=kernal_size[l_idx], initializer=layers.xavier_initializer_conv2d())
		knl_b = tf.Variable(tf.zeros([kernal_size[l_idx][-1]]))
		last_layer = tf.nn.bias_add(tf.nn.conv2d(last_layer, knl_w, strides=stride, padding='SAME'),\
			knl_b)
		last_layer = tf.nn.relu(last_layer)
	
	return last_layer, X

def gen_nlayer_conv_decoder(X, decoder_para):
	'''
		generate a n-layer convolutional encoder

        @encoder_para: dict, parameters of method, including:
            @@kernal_size: list, shape=[n:4], [[height, width, output_channel, input_channel], ... ];
				notice that elements in kernal_size has the same order with that in encoder. That is to say
				input_channel of encoder is the output_channel of decoder.
            @@strides: list, shape=[n:4], [[batch_stride, height_stride, width_stride, channel_stride], ...]
            @@output_size: list, shape=[4:], [batch, height, width, channel]
	'''

	kernal_size = decoder_para['kernal_size']
	strides = decoder_para['strides']
	output_size = decoder_para['output_size']
	
	last_layer = X
	n_layers = len(kernal_size)
	for l_idx in xrange(n_layers):
		knl_w = tf.get_variable('dec_w_%d'%l_idx, shape=kernal_size[n_layers - l_idx - 1],\
		 initializer=layers.xavier_initializer_conv2d())
		knl_b = tf.Variable(tf.zeros([kernal_size[n_layers - l_idx - 1][-2]]))
		stride = strides[n_layers - l_idx - 1]

		#calculate the output shape
		last_ly_shape = last_layer.get_shape().as_list()
		out_height = last_ly_shape[1] * stride[1] if l_idx!=n_layers-1 else output_size[1]
		out_width = last_ly_shape[2] * stride[2] if l_idx!=n_layers-1 else output_size[2]
		out_channel = kernal_size[n_layers - l_idx - 1][-2] if l_idx!=n_layers-1 else output_size[3]

		#generate last layer
		last_layer = tf.nn.bias_add(tf.nn.conv2d_transpose(last_layer, knl_w, \
		tf.stack([output_size[0], out_height, out_width, out_channel]), strides=stride, padding='SAME'),\
		knl_b)
		last_layer = tf.nn.relu(last_layer)
	
	return last_layer
