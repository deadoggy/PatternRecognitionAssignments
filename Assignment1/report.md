# Assignment 1 Report
### from 1831604 Zhang Yinjia

## 1. Dataset Description

The two datasets selected from UCI  are [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation) and [Bank Marketing](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 

The first dataset is categorical with 6 attributes. The details of these attributes are listed in the
following table:

attribute_name|values|type
-----|------|----
buying|"vhigh", "high", "med", "low".|categorical
maint|"vhigh", "high", "med", "low".|categorical
doors|"2", "3", "4", "5more".|categorical
persons|"2", "4", "more".|categorical
lug_boot|"small", "med", "big".|categorical
safety|"low", "med", "high".|categorical

All records are classified into four classes: `unacc, acc, good and vgood`. In this assignment, we treat `unacc` as `0` and others as `1`. 
And the size of the [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation) is 1728.

Another dataset [Bank Marketing](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing) is much more 
complex. The size of it is 45211 with 20 attributes, and they contains both categorical and numeric
attributes. The details are listed in the following:

attribute_name|values|type
-----|------|----
age||numeric
job|"admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"|categorical
marital|"divorced","married","single","unknown"|categorical
education|"basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"|categorical
default|"no","yes","unknown"|categorical
housing|"no","yes","unknown"|categorical
loan|"no","yes","unknown"|categorical
contact|"cellular","telephone"|categorical
month|"jan", "feb", "mar", ..., "nov", "dec"|categorical
day_of_week|"mon","tue","wed","thu","fri"|categorical
duration||numeric
campaign||numeric
pdays||numeric
previous||numeric
poutcome|"failure","nonexistent","success"|categorical
emp.var.rate||numeric
cons.price.idx||numeric
cons.conf.idx||numeric
euribor3m||numeric
nr.employed||numeric

And all data is classified into two classes by whether the client 
subscribed a term deposit.

The details of all attributes of these two dataset is shown in `./dataset/car.c45-names` and `./dataset/bank-additional-names.txt`


## Data Preprocessing

### Step1. Categorical Data
    
For all categorical data, integer type number is used to represent the data. For example,  `buying` attribute, whose values
are "vhigh", "high", "med", "low", is represented by  3, 2, 1 and 0. 

### Step2. Max-Min Scale

For both categorical and numeric data, Max-Min Scale is used to restrict values into [0, 1]. The reason is that the ranges of 
numeric data are much different from each other, in this case the initial value of `beta=<W;b>` may inflect the result of 
logistic regression. 

### Step3. Divide Dataset into K Partitons for Cross Validation

To make full use of dataset, k-fold cross validation is used to evaluate accuracy of logistic regression
model. In this experiment, the dataset is divided into 10 partitions. For each partition, the other 
partions are treated as training data and test experiment is processed on this partition. The final 
accuracy is calculated as the average accuracy of all these experiments.

## Modules of Source Code

The code is divided into three parts: `dataprocess.py`, `lr.py` and `experiments.py`

### dataprocess.py

The code in this file is to preprocess the original data into vector data. The main logic of preprocessing
is shown in `Data Description` section. The output of this module is `car.json` and `bank.json`, whose 
structure is:
```json
 {
     'X_0': [[],[],...],
     'X_1': [[],[],...]
 }
```
The `X_0` is the data whose label is `0` and the `X_1` is the data whose label is `1`.

### lr.py

This module contains the code of logistic regression. The function is organized as class 
LogisticRegression. This class contains two functions: `fit` and
`predict`. `fit` is used to train model using test data. The first part
of `fit` is initialization. The code is shown as follows:
```python
    #init
    X = np.matrix(np.hstack((X, np.ones((X.shape[0],1))))).T # shape: [n_features, n_sample]
    y = np.matrix(y).T # shape: [n_samples, 1]
    d = X.shape[0]
    p_1_func = lambda X, beta: 1/(1+np.exp(-X.T*beta))
    self._beta = np.matrix(np.zeros((d,1))) if self._beta is None else self._beta
    if self._beta.shape[0] != d:
        raise Exception('beta dimension error')
```

X here is the data matrix. First we add `1` to each vector and transform
it into shape [n_features, n_samples]. `y` is the ground truth of data. 
`d` is the dimension of data, which is `n_features`. `p_1_function` is the
lambda function to calculate $\frac{1}{1+e^{-\beta .Tx}}$ for all data 
`X`.

The second part of `fit` is iterations of newton method. The code is as follows:
```python
#newton iteration
    itrs = 0
    while itrs < self._max_itr:
        itrs += 1
        p_1 = p_1_func(X, self._beta)
        df = -1 * X*(y-p_1)
        ddf = np.matrix(np.zeros((d,d)))
        for i in xrange(X.shape[1]):
            ddf += (p_1[i,:]*(1-p_1[i,:]))[0,0] * X[:,i] * X[:,i].T 
        diff = np.linalg.pinv(ddf) * df
        if np.linalg.norm(diff) < self._tol:
            break
        self._beta -= diff
```
`itrs` is the max times of iterations. First the $\frac{1}{1+e^{-\beta .Tx}}$ 
is calculated by `p_1_function`. `df = -1 * X*(y-p_1)` is to calculate 
first derivative of `X` and 
```python
    ddf = np.matrix(np.zeros((d,d)))
    for i in xrange(X.shape[1]):
        ddf += (p_1[i,:]*(1-p_1[i,:]))[0,0] * X[:,i] * X[:,i].T
```
is to calculate second derivative of `X`. The step in newton method is
calculated by `diff = np.linalg.pinv(ddf) * df`. If the matrix `ddf` can
not be inversed, the pseudo inverse matrix of `ddf` is calculated instead.
When the step is less than `self._tol`, the iteration will be interrupted.

The `predict` method is to predict a data record. Before invoking predict,
the `fit` method must be invoked to training the model. It can be chosen 
whether return labels or probilities.

### experiments.py

There are three parts in `experiments.py`. The first part is loading data 
from  json file and scale data into range [0, 1]. The second part is using
`PCA` to reduce data to 2 dimensions. Then all the dataset is partitioned
into 10 parts equally.
