#coding:utf-8

import numpy as np
import random

class KMeans:

    def fit(self, X, k):
        '''
            KMeans on X, dividing into k clusters

            @X: np.ndarray; shape = [n_samples, n_features]
            @k: int

            #return: np.ndarray, shape = [n_samples,]
        '''
        centers = random.sample(X,k)
        y_predict = np.array([-1 for i in xrange(X.shape[0])])

        while True:
            cls_mean = np.array([ [0. for j in xrange(X.shape[1])] for i in xrange(k)])
            cls_count = np.array([0 for i in xrange(k)])
            for i, v in enumerate(X):
                # find the nearest center
                nrst_ctr = 0
                for j, c in enumerate(centers):
                    if np.sum(np.sqrt((v-c)**2)) < np.sum(np.sqrt((v-centers[nrst_ctr])**2)):
                        nrst_ctr = j
                # update y_predict, cls_mean and cls_count
                y_predict[i] = nrst_ctr
                cls_mean[nrst_ctr] += v
                cls_count[nrst_ctr] += 1
            for d in xrange(X.shape[1]):
                cls_mean[:, d] /= cls_count
            
            if np.sum(centers==cls_mean) == cls_mean.shape[0] * cls_mean.shape[1]:
                break
            else:
                centers = cls_mean

        return y_predict