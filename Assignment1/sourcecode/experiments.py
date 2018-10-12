#coding:utf-8

import sys
import numpy as np
import json
from lr import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

#load data and config experiments
################################################################################
data_path = sys.path[0] + '/../dataset/%s.json'
data_name = 'car'
with open(data_path%data_name) as data_f:
    data = json.load(data_f)
dimension = len(data[data.keys()[0]][0])

#config experiments
components = dimension #components when dimension reduction
cv_k = 10 #k in k-fold cross validation
betas = 5 #number of random initial beta(include [0., 0., ...])

#generate k-fold cross validation set
################################################################################
k_X_set = [[] for i in xrange(cv_k)]
k_y_set = [[] for i in xrange(cv_k)]
for i in xrange(len(data.keys())):
    label = data.keys()[i]
    entire_set = data[label]
    k_size = int(len(entire_set)/cv_k)
    for j, vec in enumerate(entire_set):
        set_index = int(j/k_size)
        if set_index >= cv_k:
            set_index -= 1
        k_X_set[set_index].append(vec)
        k_y_set[set_index].append(i)

#dimensionality reduction using LDA
###############################################################################
dim_reducer = PCA(n_components=components) if components!=dimension else None
for si in xrange(cv_k):
    k_y_set[si] = np.array(k_y_set[si])
    k_X_set[si] = dim_reducer.fit_transform(k_X_set[si]) if dim_reducer is not None else np.array(k_X_set[si])

#run experiments by cross validation
###############################################################################
initial_beta = ((np.random.random_sample(size=(betas-1, components+1))-0.5)*4).tolist() if betas > 1 else []
initial_beta.append([0. for i in xrange(components+1)])

best_initial_beta = None
best_rate = -0.1
for beta in initial_beta:
    tmp_rate = 0.
    for i, test_X in enumerate(k_X_set):
        test_y = k_y_set[i]
        #generate train_X
        train_X = []
        train_y = []
        for j in xrange(len(k_X_set)):
            if j==i:
                continue
            train_X.append(k_X_set[j])
            train_y.append(k_y_set[j])
        train_X = np.vstack(train_X)
        train_y = np.hstack(train_y)
        lr = LogisticRegression(beta=beta)
        lr.fit(train_X, train_y)
        exp_y = lr.predict(test_X)
        tmp_rate += 1.0-(exp_y^test_y).sum()/float(len(test_y))
    tmp_rate /= cv_k
    if tmp_rate > best_rate:
        best_initial_beta = beta
        best_rate = tmp_rate

print best_initial_beta
print best_rate

        


