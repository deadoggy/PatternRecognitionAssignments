#coding:utf-8

from hac import Hierarchical, SingleLinkage, CompleteLinkage, AverageLinkage
from kmeans import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
import sys
import json
import numpy as np


dataset = "iris"
with open("%s/../dataset/%s.json"%(sys.path[0], dataset), "r") as dataset_in:
    data_json = json.load(dataset_in)

X = []
y_truth = []

for key in data_json.keys():
    X.extend(data_json[key])
    y_truth.extend([int(key) for i in xrange(len(data_json[key]))])

X = np.array(X)
y_truth = np.array(y_truth)

kmeans_json = {
        "mse":[],
        "sc":[],
        "ari":[]
}
hac_json = {
        "Avg": {
            "mse":[],
            "sc":[],
            "ari":[]
        },
        "Sgl": {
            "mse":[],
            "sc":[],
            "ari":[]
        },
        "Cpl": {
            "mse":[],
            "sc":[],
            "ari":[]
        }
}


def calc_ctr(X, y_predict, k):
    ctrs = [np.array([0. for d in xrange(X.shape[1])]) for i in xrange(k)]
    for i,x in enumerate(X):
        ctrs[y_predict[i]] += x
    for i,c in enumerate(ctrs):
        c /= y_predict.tolist().count(i)
    y_c = []
    for i,x in enumerate(X):
        y_c.append(ctrs[y_predict[i]])
    return y_c

def Mse(X, y_predict, k):
    y_ctrs = calc_ctr(X, y_predict, k)
    return mean_squared_error(X, y_ctrs)


print "KMeans"
print "============================"
for k in xrange(2, 10):
    print "k=%d"%k
    #kmeans
    kmeans_predict = KMeans().fit(X, k)
    mse = Mse(X, kmeans_predict, k)
    sc = silhouette_score(X, kmeans_predict)
    ari = adjusted_rand_score(y_truth, kmeans_predict)
    kmeans_json["sc"].append(sc)
    kmeans_json["mse"].append(mse)
    kmeans_json["ari"].append(ari)

with open("%s/../kmeans_%s_out.json"%(sys.path[0], dataset), "w") as kmeans_out:
    json.dump(kmeans_json, kmeans_out)

# print "Hac"
# print "============================"
# linkage_names = ['Avg','Cpl','Sgl']
# for lidx, linkage in enumerate([AverageLinkage, CompleteLinkage, SingleLinkage]):
#     linkage_name = linkage_names[lidx]
#     print "linkage=%s"%linkage_name
#     print "----------------"
#     for k in xrange(2, 10):
#         print "k=%d"%k
#         hac_predict = Hierarchical().fit(X, k, linkage)
#         mse = Mse(X, hac_predict, k)
#         sc = silhouette_score(X, hac_predict)
#         ari = adjusted_rand_score(y_truth, hac_predict)
#         hac_json[linkage_name]["sc"].append(sc)
#         hac_json[linkage_name]["mse"].append(mse)
#         hac_json[linkage_name]["ari"].append(ari)

# with open("%s/../hac_%s_out.json"%(sys.path[0], dataset), "w") as hac_out:
#     json.dump(hac_json, hac_out)
