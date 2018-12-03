#coding:utf-8

import numpy as np

class Cluster:
    def __init__(self, first_one=None):
        '''
            init function of Cluster

            @first_one: (val, idx), the initial member of Cluster
        '''
        self.members = []
        self.mem_idx = []
        if first_one is not None:
            self.members.append(first_one[0])
            self.mem_idx.append(first_one[1])

    def merge(self, other_cls):
        '''
            merge this Cluster and other_cls

            @other_cls: Cluster
        '''
        self.members.extend(other_cls.members)
        self.mem_idx.extend(other_cls.mem_idx)

class Hierarchical:

    def fit(self, X, k, linkage):
        '''
            hierarchical clustering on X, dividing into k clusters
        
            @X: np.ndarray, shape=[n_samples, n_features]

            @k: int, number of clusters

            @linkage: callable, given two Cluster obj, return the linkage between them

            #return: np.ndarray, shape=[n_samples,]
        '''
        if type(X) != np.ndarray:
            raise Exception('X not a numpy.ndarray')

        clus = [ Cluster((X[i],i)) for i in xrange(X.shape[0])]
        adj_mat = np.matrix(np.zeros((X.shape[0], X.shape[0])))

        #initialize adjacent matrix
        for i in xrange(len(clus)):
            for j in xrange(i, len(clus)):
                dist = linkage(clus[i], clus[j]) if i!=j else float('inf')
                adj_mat[i,j] = adj_mat[j,i] = dist
        
        #run merge
        while True:
            min_dist = adj_mat.min()
            min_loc = np.where(adj_mat == min_dist)
            x = min_loc[0][0]
            y = min_loc[1][0]
            clus_1 = clus[x]
            clus_2 = clus[y]
            clus_1.merge(clus_2)
            clus.remove(clus_2)
            # delete min_loc[1]-th row and col
            adj_mat = np.delete(adj_mat, y, axis=0)
            adj_mat = np.delete(adj_mat, y, axis=1)

            for i in xrange(len(clus)):
                dist = linkage(clus[i], clus_1) if i!= x else float('inf')
                adj_mat[i, x] = dist
                adj_mat[x, i] = dist
            if len(clus) == k:
                break
        
        #return label
        y_predict = np.array([-1 for i in xrange(X.shape[0])])
        for i, c in enumerate(clus):
            for l in c.mem_idx:
                y_predict[l] = i
        return y_predict


def SingleLinkage(cls_1, cls_2):
    '''
        single linkage between cls_1 and cls_2

        @cls_1/cls_2: Cluster
    '''
    dist = float('inf')
    for m1 in cls_1.members:
        for m2 in cls_2.members:
            tmp_dist = np.sum(np.sqrt( (m1-m2)**2 ))
            dist = tmp_dist if tmp_dist < dist else dist
    return dist

def CompleteLinkage(cls_1, cls_2):
    '''
        complete linkage between cls_1 and cls_2

        @cls_1/cls_2: Cluster
    '''
    dist = -float('inf')
    for m1 in cls_1.members:
        for m2 in cls_2.members:
            tmp_dist = np.sum(np.sqrt( (m1-m2)**2 ))
            dist = tmp_dist if tmp_dist > dist else dist
    return dist

def AverageLinkage(cls_1, cls_2):
    '''
        average linkage between cls_1 and cls_2

        @cls_1/cls_2: Cluster
    '''
    dist = 0.
    for m1 in cls_1.members:
        for m2 in cls_2.members:
            dist += np.sum(np.sqrt( (m1-m2)**2 ))
    return dist / (len(cls_1.members)*len(cls_2.members))