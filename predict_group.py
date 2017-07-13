# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:07:52 2017

@author: Bouamama Amine


"""


import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error, auc, roc_curve
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime

np.random.seed(2017)

""" ====== LOGGING ========== """


""" ========================= """



""" define working directory"""
os.chdir("C:/Documents/4_ML Ressources/Kaggle/Zillow")



def read_data(train_or_test = 'train'):
    if((train_or_test == 'train' )| (train_or_test == 'test')):
        df = pd.read_csv(
                "data/s1_intermediate/"+train_or_test+".csv"
                #parse_dates=['transactiondate']
                )
    else : 
        raise Exception("Do you want to read train or test ?")
        
    df.set_index('parcelid',inplace=True)
    return df
    

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

def describe_features(df) :
    data_desc = pd.read_excel("data/s0_raw/zillow_data_dictionary.xlsx")
    cols = df.columns.values
    return data_desc[data_desc.Feature.isin(cols)]
    

def feature_engineering(df,verbose=False) :
    
    # remove date column
    if 'transactiondate' in df.columns:
        df.drop('transactiondate',axis=1,inplace=True)
    
    # remove fips
    if 'fips' in df.columns:
        df.drop('fips',axis=1,inplace=True)
    
    # remove propertycountylandusecode
    if 'propertycountylandusecode' in df.columns:
        df.drop('propertycountylandusecode',axis=1,inplace=True)
        
    # remove censustractandblock
    if 'censustractandblock' in df.columns:
        df.drop('censustractandblock',axis=1,inplace=True)
    
        # remove censustractandblock
    if 'rawcensustractandblock' in df.columns:
        df.drop('rawcensustractandblock',axis=1,inplace=True)
             
    # remove regionidzip
    if 'regionidzip' in df.columns:
        df.drop('regionidzip',axis=1,inplace=True)
    
    # remove regionidcounty
    if 'regionidcounty' in df.columns:
        df.drop('regionidcounty',axis=1,inplace=True)
        
    # remove regionidcity
    if 'regionidcity' in df.columns:
        df.drop('regionidcity',axis=1,inplace=True)
    
    
        
    # fill nas with mean
    for ft in df.columns :
        df[ft] = df[ft].fillna(df[ft].mean())

    # remove regionidcity
    if 'logerror' in df.columns:
        q05 = df.logerror.quantile(0.05)
        q95 = df.logerror.quantile(0.95)
        df['group'] = 0
        df.loc[df.logerror < q05,"group"] = -1
        df.loc[df.logerror > q95,"group"] = 1
        df.drop('logerror',axis=1,inplace=True)
    
    if verbose :
        print(df.dtypes)
        
    return df

def create_inputs_model(df,test_size=0.3):
    
    X,y = df.drop('group',axis=1).values, df['group'].values
    
    # create train and test set
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size) 
    
    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
        
    return X_train, X_test, y_train, y_test, scaler
    

def train_model(X_train,y_train):
    print("training the model ...")
    rf = RandomForestClassifier(n_estimators=1000,
                               max_depth=8,
                               n_jobs=-1,
                               verbose=1)
#    rf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0,probability=True)
    
    rf.fit(X_train,y_train)
    y_pred_train = rf.predict_proba(X_train)
    
    
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_train[:,0], pos_label=1)
    print("AUC on train : {:.02f} %".format( auc(fpr, tpr)*100))
    
    return rf

def evaluate_model(model, X_val, y_val,plot=False):
    print("="*40)
    print("Model Evaluation")
    
    y_pred = model.predict_proba(X_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_pred[:,0], pos_label=1)
    print("AUC on test : {:.02f} %".format( auc(fpr, tpr)*100))
       
    print("\n")
    if plot:
        plt.scatter(fpr,tpr,s=1,alpha=0.7)
#        plt.plot(y_pred,y_pred,alpha=0.7,c='r')
    return y_pred



def dummy_trainer(X_train,y_train):
    print("training the dummy model ...")
    y_pred_train = np.median(y_train)
    return y_pred_train
    
def evaluate_dummy_model(y_pred_train, X_val, y_val,y_train=None,plot=False):
    print("="*40)
    print("Model Evaluation")
    
#    y_pred = np.ones((y_val.shape[0],1)) * y_pred_train
    
    q10 = np.percentile(y_train,25)
    q90 = np.percentile(y_train,75)
    y_pred = np.array([(y if ((y<=q10) | (y>=q90) ) else y_pred_train )
                for y in y_val])
                     
    print(".. test RMSE : {:0.3f} %".format(mean_squared_error(y_val,y_pred)*100))
    print(".. test MAE  : {:0.3f} %".format(mean_absolute_error(y_val,y_pred)*100))
    
    print("\n")
    if plot:
        plt.scatter(y_pred,y_val,s=1,alpha=0.7)
        plt.plot(y_pred,y_pred,alpha=0.7,c='r')
    return y_pred



""" ============= MAIN ============== """

## read data
df           = read_data()
## select only relevant features
cols_to_keep = select_most_present_features(df,load=True)
df           = df[cols_to_keep]

## feature engineering
df           = feature_engineering(df)
  

## create train and test set
X_train, X_val, y_train, y_val, scaler = create_inputs_model(df,test_size = 0.25)


## train model
mdl         = train_model(X_train,y_train)
#mdl         = dummy_trainer(X_train,y_train)

## predict on val set and evaluate
y_val_pred  = evaluate_model(mdl, X_val,y_val,plot=True)
#y_val_pred  = evaluate_dummy_model(mdl, X_val,y_val,y_train)
