#coding:utf-8

import sys
import numpy as np
import json
from sklearn import 


data_folder = sys.path[0] + '/../dataset'

with open(data_folder + '/car.data') as cardata_f:
    lines = cardata_f.read().split('\n')

car_data = {
    'X_1':[],
    'X_0':[]
}

car_dict = {
    'vhigh':3,
    'high':2,
    'big':2,
    'med':1,
    'low':0,
    'small':0,
    '2':2,
    '3':3,
    '4':4,
    '5more':6,
    'more':6,
    'unacc':0,
    'acc':1,
    'good':1,
    'vgood':1
}

for line in lines:
    if ''==line:
        continue
    vals = line.split(',')
    rec = []
    for i in xrange(len(vals)-1):
        rec.append(car_dict[vals[i]])
    label = car_dict[vals[-1]]

    if 1==label:
        car_data['X_1'].append(rec)
    else:
        car_data['X_0'].append(rec)
        
with open(data_folder + '/car.json', 'w') as car_out:
    json.dump(car_data, car_out)

#bank
with open(data_folder + '/bank-additional-full.csv') as bank_f:
    lines = bank_f.read().split('\r\n')

bank_data = {
    'X_0':[],
    'X_1':[]
}

bank_dict = {
    '"admin."':0,
    '"blue-collar"':1,
    '"entrepreneur"':2,
    '"housemaid"':3,
    '"management"':4,
    '"retired"':5,
    '"self-employed"':6,
    '"services"':7,
    '"student"':8,
    '"technician"':9,
    '"unemployed"':10,
    '"unknown"':-1,
    '"divorced"':0,
    '"married"':1,
    '"single"':2,
    '"basic.4y"':0,
    '"basic.6y"':1,
    '"basic.9y"':2,
    '"high.school"':3,
    '"illiterate"':4,
    '"professional.course"':5,
    '"university.degree"':6,
    '"yes"':1,
    '"no"':0,
    '"cellular"':0,
    '"telephone"':1,
    '"failure"':0,
    '"nonexistent"':1,
    '"success"':2
}

# 9, 10 ignore
for i, line in enumerate(lines):
    if ''==line or i==0:
        continue
    vals = line.split(';')
    rec = []
    for i in xrange(len(vals)-1):
        if i == 8 or i == 9:
            continue
        if '"' in vals[i]:
            rec.append(bank_dict[vals[i]])
        else:
            rec.append(float(vals[i]))

    label = bank_dict[vals[-1]]

    if 1==label:
        bank_data['X_1'].append(rec)
    else:
        bank_data['X_0'].append(rec)

with open(data_folder + '/bank.json', 'w') as bank_out:
    json.dump(bank_data, bank_out)

