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

