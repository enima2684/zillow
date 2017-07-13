# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 19:38:15 2017

@author: Bouamama Amine

Form train and test data
"""

import os
import pandas as pd
os.chdir("C:\Documents/4_ML Ressources/Kaggle/Zillow")

# read raw data
properties = pd.read_csv('data/s0_raw/properties_2016.csv')
train_raw  = pd.read_csv('data/s0_raw/train_2016_v2.csv')


""" construct training set """
train  = train_raw.merge(properties,on='parcelid',how='left')
train.to_csv('data/s1_intermediate/train.csv',index=False)

""" construct test set """
train_ids = train.parcelid.values
test = properties

# check that we have unique ids
assert(len(test.parcelid.unique()) == len(test)) 

# write the test table
test.to_csv('data/s1_intermediate/test.csv',index=False)


""" comparing train and test """
print("% of articles in training set : {:.2f} %".format(len(train) / (len(train) + len(test))*100))
