#coding:utf-8

import sys
import numpy as np
import json
from lr import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, MinMaxScaler

#load data, config experiments, scale and dimension reduction
################################################################################

#load
data_path = sys.path[0] + '/../dataset/%s.json'
data_name = 'car'
with open(data_path%data_name) as data_f:
    data = json.load(data_f)
dimension = len(data[data.keys()[0]][0])

#config experiments
components = 2 #components when dimension reduction using PCA
cv_k = 10 #k in k-fold cross validation
betas = 5 #number of random initial beta(include [0., 0., ...])

# scale
l1_len = len(data['X_1'])
minmaxscaler = MinMaxScaler()
scale_array = minmaxscaler.fit_transform(np.array(data['X_1'] + data['X_0']))

#dimension reduction
dim_reducer = PCA(n_components=components, svd_solver='full') if components!=dimension else None
if dim_reducer is not None:
    scale_array = dim_reducer.fit_transform(scale_array)
data['X_1'] = scale_array[0:l1_len]
data['X_0'] = scale_array[l1_len:]

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


for i in xrange(len(k_X_set)):
    k_X_set[i] = np.array(k_X_set[i])
    k_y_set[i] = np.array(k_y_set[i])

#run experiments by cross validation
###############################################################################
initial_beta = ((np.random.random_sample(size=(betas-1, components+1))-0.5)*4).tolist() if betas > 1 else []
initial_beta.insert(0, [0. for i in xrange(components+1)])

best_initial_beta = None
best_rate = -0.1
best_label = None
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

lr = LogisticRegression(beta=best_initial_beta)
X = []
y = []
for i in xrange(cv_k-1):
    X.append(k_X_set[i])
    y.append(k_y_set[i])
X = np.vstack(X)
y = np.hstack(y)
lr.fit(X, y)
label = lr.predict(k_X_set[cv_k-1])
rate = 1.0-(k_y_set[cv_k-1]^label).sum()/float(len(label))
out_json = {
    'X': k_X_set[cv_k-1].tolist(),
    'predict_y': label.tolist(),
    'truth_y': k_y_set[cv_k-1].tolist(),
    'beta': lr._beta.T.tolist(),
    'rate': rate
}

with open(sys.path[0] + '/../dataset/%s.out'%data_name, "w") as out:
    json.dump(out_json, out)

        


