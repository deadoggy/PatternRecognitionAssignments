#coding:utf-8

from __future__ import division, absolute_import
import tensorflow as tf
import numpy as np
from sklearn import cluster
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import sys
from Conv_EncDec import gen_conv_decoder, gen_conv_encoder
from DSC_Net import DSC_Net
from sklearn.metrics.cluster import adjusted_rand_score
from munkres import Munkres



data_dict = sio.loadmat('/home/yinjia/Documents/Deep-subspace-clustering-networks/Data/COIL20.mat')
imgs = np.reshape(data_dict['fea'], [len(data_dict['fea']), 32, 32, 1])
ground_truth = data_dict['gnd']

def threshold_weightmat(W, rate):
    '''
        a threshold of W. Leave the least singlar values whose sum > rate * (sum of row/col)

        @W: the weight mat
        @rate: rate of sum
    '''
    
    if rate >= 1.:
        return W
    
    rlt = np.zeros(W.shape)
    sorted_by_col = np.abs(np.sort(-np.abs(W), axis=0))
    sorted_by_col_idx = np.argsort(-np.abs(W), axis=0)
    for c in xrange(W.shape[1]):
        col_sum = np.sum(sorted_by_col[:, c]).astype(float)
        r = 0
        tmp_sum = 0.
        while tmp_sum <= rate * col_sum:
            tmp_sum += sorted_by_col[r, c]
            r = r + 1
        rlt[sorted_by_col_idx[0:r+1, c], c] = sorted_by_col[sorted_by_col_idx[0:r+1, c], c]
    
    return rlt

def generate_affinity_mat(W, k, d, alpha):
    '''
        generate affinity matrix by：
         'P. Ji, M. Salzmann, and H. Li. Efficient dense subspace clustering. In WACV, pages 461–468. IEEE, 2014'
        
        @W: weight matrix
        @k: number of clusters
        @d: maximal intrinsic dimension of the subspaces
        @alpha: empirically selected according to the level of noise
    '''

    m = k * d + 1
    W = .5 * (W + W.T) # make sure W is a sym mat in case of precision errors
    U, Sigma, VT = svds(W, m)
    U = U[:,::-1] # reserves the order of U because return of svds is in descend order
    Sigma = np.diag(np.sqrt(Sigma[::-1]))
    Z = normalize(U.dot(Sigma), norm='l2', axis=1)
    Z = Z.dot(Z.T)
    Z *= (Z>0) # if a value<0, then set value=>0
    L = np.abs(Z ** alpha)
    L /= L.max()
    L = .5 * (L + L.T)
    return L

def ConvEncDec_Exp():
    img_size = [None, 32, 32, 1]
    kernal_size = [3, 3, 1, 15]
    strides = [1, 2, 2, 1]
    reg1 = 1.0
    reg2 = 150.0
    lr = 1e-3

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
    
    print "spectral clustering"
    threshold_rate = 0.04
    k = 20
    # W = thrC(weight_mat, threshold_rate)
    # affinity_mat = post_proC(W, k, 12, 8)

    W = threshold_weightmat(weight_mat, threshold_rate)
    affinity_mat = generate_affinity_mat(W, k, 12, 8)

    spec = cluster.SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize')
    y_predict = spec.fit_predict(affinity_mat) + 1
    adj_rd_idx = adjusted_rand_score(ground_truth.T[0], y_predict)
    return adj_rd_idx
    

print ConvEncDec_Exp()

