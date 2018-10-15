#coding:utf-8

import sys
import numpy as np
import json

spliter = ','
data_name = 'iris'
label_0 = 1
label_1 = 2
data_path = sys.path[0] + '/../dataset/%s.data'
X0_lim = 5000
X1_lim = 5000
with open(data_path%data_name) as haber_data_f:
    lines = haber_data_f.read().split('\n')

haber_data = {
    'X_1':[],
    'X_0':[]
}

for line in lines:
    if ''==line:
        break
    vals = line.split(spliter)
    vec = [float(vals[0]), float(vals[2]), float(vals[3])]
    label = int(vals[4])
    if label==label_0 and len(haber_data['X_0']) < X0_lim:
        haber_data['X_0'].append(vec)
    elif label==label_1 and len(haber_data['X_1']) < X1_lim:
        haber_data['X_1'].append(vec)

out_path = sys.path[0] + '/../dataset/%s.json'
with open(out_path%data_name, 'w') as out:
    json.dump(haber_data, out)
