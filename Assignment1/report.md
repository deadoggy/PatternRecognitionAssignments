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

All records are classified into four classes: <strong>unacc, acc, good and vgood</strong>. And the size of 
the [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation) is 1728.

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


