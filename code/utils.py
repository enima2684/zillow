# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:57:49 2017

@author: Bouamama Amine

utils functions used in the zillow project
"""

import pandas as pd
import yaml
import os
import pickle
import winsound

""" define working directory"""
os.chdir("C:/Documents/4_ML Ressources/Kaggle/Zillow")

def read_params(step):
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg[step]


def read_data(train_or_test = 'train'):
    if(train_or_test == 'train'):
        df = pd.read_csv(
                "data/s1_intermediate/"+train_or_test+".csv",
                parse_dates=['transactiondate']
                )
    elif (train_or_test == 'test'):
        df = pd.read_csv(
                "data/s1_intermediate/"+train_or_test+".csv"
                )
    else : 
        raise Exception("Do you want to read train or test ?")
        
#    df.set_index('parcelid',inplace=True)
    return df

def read_bins(df):
    bins = pd.read_csv('data/s1_intermediate/train_bins.csv')
    df   = df.merge(bins,on='parcelid')
    return df

def describe_features(df) :
    data_desc = pd.read_excel("data/s0_raw/zillow_data_dictionary.xlsx")
    cols = df.columns.values
    return data_desc[data_desc.Feature.isin(cols)]  



def select_most_present_features(df,load = True,verbose = False,threshold = 0.6,save = False):
    
    if load :
        with open("data/s2_meta/most_present_features.pkl","rb") as fp :
            cols_to_keep =  pickle.load(fp)
    
    
    if save :
        threshold *= len(df)
        cols_to_keep = [ col for col in df.columns.values if len(df[~df[col].isnull()]) > threshold ]
        with open("data/s2_meta/most_present_features.pkl","wb") as fp :
            pickle.dump(cols_to_keep,fp)
    
    if verbose:
        print("Selecting {} features.".format(len(cols_to_keep)))
    return cols_to_keep

def read_classification_output():
    class_output = pd.read_csv('data/s1_intermediate/test_bin_prediction.csv')
    return class_output

def read_logerror_per_bin():
    logerror_per_bin = pd.read_csv('data/s1_intermediate/train_logerror_per_bin.csv')
    return logerror_per_bin

def alert(nb_dongs=3):
    for i in range(nb_dongs):
        winsound.PlaySound('media/warning.wav', winsound.SND_FILENAME)  
