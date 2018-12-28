#coding:utf-8



from __future__ import division, absolute_import
import tensorflow as tf
import numpy as np
from sklearn import cluster


class DSC_Net:

    def __init__(self, encoder, encoder_para, decoder, decoder_para, batch_size, x_dim,\
        reg_rate_1=1.0, reg_rate_2=1.0):
        '''
            init function of a DSC_Net

            @encoder: callable, a func to generate encoder
            @encoder_para: dict, parameter of encoder
            @decoder: callable, a func to generate decoder
            @decoder_para: dict, parameter of decoder
            @batch_size: int, size of input data
            @x_dim: np.ndarray
            @itr: int, iteration times
            @reg_rate_1: float, rate of ||C||^2 in loss
            @reg_rate_2: float, rate of ||ZC-Z||^2 in loss
            @output_path: str, model output path
            @log_path: str, log output path
        '''

        # assignment parameters
        self.encoder, self.encoder_input_pl = encoder(encoder_para)
        self.batch_size = batch_size
        self.reg_rate_1 = reg_rate_1
        self.reg_rate_2 = reg_rate_2
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.itr = 0
        self.x_dim = x_dim
        self.log_path = '/log/'
        
        # self expressiveness layer
        self.weight_mat = tf.Variable(tf.ones([self.batch_size, self.batch_size], tf.float32)*1.0e-8)
        self.Z = tf.reshape(self.encoder, [batch_size, -1])
        self.Z_C = tf.matmul(self.weight_mat, self.Z)
        
        latent = tf.reshape(self.Z_C, tf.shape(self.encoder))
        self.decoder = decoder(latent, decoder_para)


        # calculate loss
        self.weight_loss = tf.reduce_sum(tf.pow(self.weight_mat, 2.)) * self.reg_rate_1
        self.selfexp_loss = tf.reduce_sum(tf.pow(tf.subtract(self.Z_C, self.Z), 2.)) * .5 * self.reg_rate_2
        self.recover_loss = tf.reduce_sum(tf.pow(tf.subtract(self.encoder_input_pl, self.decoder), 2.))
        self.total_loss = self.recover_loss + self.weight_loss + self.selfexp_loss

        tf.summary.scalar("weight_loss", self.weight_loss)
        tf.summary.scalar("recover_loss", self.recover_loss)
        tf.summary.scalar("selfexp_loss", self.selfexp_loss)

        self.summary_op = tf.summary.merge_all()
        
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        
        #init and summary
        self.tf_session = tf.InteractiveSession()
        self.tf_session.run(tf.global_variables_initializer())
        self.summary_wtr = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())

    def initlization(self):
		tf.reset_default_graph()
		self.tf_session.run(tf.global_variables_initializer())

    def train(self, X, learning_rate, ):
        # train
        Z, weight_mat, recover_loss, weight_loss, selfexp_loss, summary_op, optimizer = \
        self.tf_session.run((self.Z, self.weight_mat, self.recover_loss, self.weight_loss, self.selfexp_loss, \
        self.summary_op, self.optimizer), feed_dict={self.encoder_input_pl:X, \
        self.learning_rate:learning_rate})
        # record
        self.summary_wtr.add_summary(summary_op, self.itr)
        self.itr = self.itr + 1
        #return
        return weight_mat, recover_loss, weight_loss, selfexp_loss

    