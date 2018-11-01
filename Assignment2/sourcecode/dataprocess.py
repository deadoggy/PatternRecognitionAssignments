#coding:utf-8

import sys
import json

#wine
# with open(sys.path[0] + "/../dataset/wine.data") as wine_f:
#     wine_lines = wine_f.read().split('\n')

# wine_data = {1:[], 2:[], 3:[]}
# for line in wine_lines:
#     if ''==line:
#         continue
#     str_vals = line.split(',')
#     vals = []
#     key = -1
#     for index, sv in enumerate(str_vals):
#         if index==0:
#             key = int(sv)
#         else:
#             vals.append(float(sv))    
#     wine_data[key].append(vals)

# with open(sys.path[0] + '/../dataset/wine.json', 'w') as wine_json:
#     json.dump(wine_data, wine_json)

#iris

with open(sys.path[0] + "/../dataset/iris.data") as iris_f:
    iris_lines = iris_f.read().split('\n')

iris_data = {'0':[], '1':[], '2':[]}
for line in iris_lines:
    if ''==line:
        continue
    str_vals = line.split(',')
    val = []
    for i in xrange(4):
        val.append(float(str_vals[i]))
    iris_data[str_vals[-1]].append(val)

with open(sys.path[0] + "/../dataset/iris.json",'w') as iris_json:
    json.dump(iris_data, iris_json)