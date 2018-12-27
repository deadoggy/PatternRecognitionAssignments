#coding:utf-8
from __future__ import division, absolute_import

import tensorflow as tf
import numpy as np
from sklearn import cluster


class DSC_Net:

    def __init__(self, encoder, decoder, decoder_input_pl, x_dim, data_size, itr, \
        reg_rate_1=1.0, reg_rate_2=1.0, output_path="/log/", log_path="/log/"):
        '''
            init function of a DSC_Net

            @encoder: an tensor of encoder
            @decoder: an tensor of decoder with symmetric structure of encoder
            @decoder_input_pl: placeholder, the input of decoder
            @x_dim: np.ndarray, dimension of input data
            @data_size: int, size of input data
            @itr: int, iteration times
            @reg_rate_1: float, rate of ||C||^2 in loss
            @reg_rate_2: float, rate of ||ZC-Z||^2 in loss
            @output_path: str, model output path
            @log_path: str, log output path
        '''

        # assignment parameters
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_input_pl = decoder_input_pl 
        self.data_size = data_size
        self.x_dim = x_dim
        self.itr = itr
        self.reg_rate_1 = reg_rate_1
        self.reg_rate_2 = reg_rate_2
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.x = tf.placeholder(tf.float32, x_dim)
        self.output_path = output_path
        self.log_path = log_path
        
        # self expressiveness layer
        self.weight_mat = tf.Variable(tf.ones([self.data_size, self.data_size], tf.float32)*1.0e-8, name='WeightMat')
        self.Z_C = tf.matmul(self.weight_mat, self.encoder)
        
        # calculate loss
        self.weight_loss = tf.reduce_sum(tf.pow(self.weight_mat, 2.)) * self.reg_rate_1
        self.recover_loss = tf.reduce_sum(tf.pow(tf.subtract(self.x, self.decoder), 2.)) * self.reg_rate_2 * .5
        self.selfexp_loss = tf.reduce_sum(tf.pow(tf.subtract(self.Z_C, self.encoder), 2.)) * .5

        tf.summary.scalar("weight_loss", self.weight_loss)
        tf.summary.scalar("recover_loss", self.recover_loss)
        tf.summary.scalar("selfexp_loss", self.selfexp_loss)

        self.total_loss = self.recover_loss + self.weight_loss + self.selfexp_loss
        self.summary_op = tf.summary.merge_all()
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        
        #init and summary
        self.tf_session = tf.InteractiveSession()
        self.tf_session.run(tf.global_variables_initializer())
        self.summary_wtr = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())

    def train(self, X, learning_rate):
        Z_C, recover_loss,  =self.tf_session.run((self.Z_C, self.recover_loss, self.weight_loss, self.selfexp_loss, \
        self.summary_op, self.optimizer), feed_dict={self.decoder_input_pl:self.Z_C, self.x:X, \
        self.learning_rate:learning_rate})