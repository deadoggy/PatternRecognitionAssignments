#coding:utf-8

import sys
import json

#wine
with open(sys.path[0] + "/../dataset/wine.data") as wine_f:
    wine_lines = wine_f.read().split('\n')

wine_data = {1:[], 2:[], 3:[]}
for line in wine_lines:
    if ''==line:
        continue
    str_vals = line.split(',')
    vals = []
    key = -1
    for index, sv in enumerate(str_vals):
        if index==0:
            key = int(sv)
        else:
            vals.append(float(sv))    
    wine_data[key].append(vals)

with open(sys.path[0] + '/../dataset/wine.json', 'w') as wine_json:
    json.dump(wine_data, wine_json)

