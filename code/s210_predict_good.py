# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:47:15 2017

@author: Bouamama Amine

model to learn the bin
"""


import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, log_loss, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import json
import gc


from utils import read_data, read_params, alert
from utils import select_most_present_features
from utils import read_bins



np.random.seed(2017)

""" define working directory"""
os.chdir("C:/Documents/4_ML Ressources/Kaggle/Zillow")

def feature_engineering(df,isTest=False,verbose=False) :

    if 'transactiondate' in df.columns:
#        # extract month of the year
#        df['Month'] = df['transactiondate'].map(lambda x: x.month)
#        df['YearMonth'] = df['transactiondate'].map(lambda x: 100*x.year + x.month)
#
#        # calulate number of properties sold during the same period
#        temp = df.groupby('YearMonth',as_index=False).parcelid.count()
#        temp.rename(columns={'parcelid':'nb_sold_same_month'},inplace=True)
#        df = df.merge(right = temp,
#                      on = 'YearMonth',
#                      how='left')


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


    # construct geographical feature

    # fill nas with mean
    for ft in df.columns :
        df[ft] = df[ft].fillna(df[ft].median())


    if verbose :
        print(df.dtypes)

    df.set_index('parcelid',inplace=True)

    
    # data selection 
    # 0 : bad fit
    # 1 : good fit
    if not isTest :
        df.loc[df.bin==2,'bin'] = 0

    return df



def create_inputs_model(df,objectif_var,test_size=0.3):

    X,y = df.drop(objectif_var,axis=1).values, df[objectif_var].values

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
    
    # create sets for probability calibration
    X_train_train, X_prob_cal, y_train_train, y_prob_cal = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.2)

    
    rf = RandomForestClassifier(
               max_features="auto",
               n_estimators=2000,
               max_depth=8,
               n_jobs=-1,
               class_weight = 'balanced',
               verbose=1)
    rf.fit(X_train_train,y_train_train)
    
    # feature importances
   
#    feature_importance = False
#    if(feature_importance):
#        
#        importances = rf.feature_importances_
#        std = np.std([tree.feature_importances_ for tree in rf.estimators_],
#                 axis=0)
#        indices = np.argsort(importances)[::-1]
#        col_names = df.drop('bin',axis=1).columns.values
#        print("Feature ranking:")
#        
#        for f in range(X_train_train.shape[1]):
#            print("%d. %s (%f)" % (f + 1, col_names[indices[f]], importances[indices[f]]))
#        
#        # Plot the feature importances of the forest
#        plt.figure()
#        plt.title("Feature importances")
#        plt.bar(range(X_train_train.shape[1]), importances[indices],
#               color="r", yerr=std[indices], align="center")
#        plt.xticks(range(X_train_train.shape[1]), col_names[indices],rotation = 50)
#        plt.xlim([-1, X_train_train.shape[1]])
#        plt.show()
        
    
    # Probability calibration
    sig_clf = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
    sig_clf.fit(X_prob_cal, y_prob_cal)
    y_pred_train = sig_clf.predict_proba(X_train)
    
    
    print(".. training log_loss  : {:0.2f} %".format(log_loss(y_train,y_pred_train)*100))
    return sig_clf

def tune_model(X,y,K=5):
    print("tuning the model ...")

    """logging"""

    # the winner is
#    {'max_features' : [sqrt'],
#              'n_estimators' : [2000],
#              'min_samples_leaf' : [1]
#              }
#
    """ """

    params = {'max_features' : ['auto','sqrt',0.2,0.4],
              'n_estimators' : [10,50,100,500,1000,2000],
              'min_samples_leaf' : [0.01,0.02,0.05,0.1,0.15,0.2],
              'max_depth' : [None,3,5,7,8,9,10]
              }

    nb_scenarios = np.product([len(params[x]) for x in params])
    results = []
    for max_f in params['max_features'] :
        for n_est in params['n_estimators'] :
            for min_leaf in params['min_samples_leaf'] :
                for max_dep in params['max_depth']:
                    kf = StratifiedKFold(n_splits=K)
                    errors_fold = []
                    for train_index, test_index in kf.split(X,y):
                        X_train_bis, X_test = X[train_index], X[test_index]
                        y_train_bis, y_test = y[train_index], y[test_index]
    
                        rf = RandomForestClassifier(
                                   max_features=max_f,
                                   n_estimators=n_est,
                                   min_samples_leaf=min_leaf,
                                   max_depth=max_dep,
                                   n_jobs=-1,
                                   class_weight = 'balanced')
                        rf.fit(X_train_bis,y_train_bis)
                        y_pred_test = rf.predict_proba(X_test)
                        logloss = log_loss(y_test,y_pred_test)
                        errors_fold.append(logloss)


                    result = {'max_features' : max_f,
                              'n_estimators':n_est,
                              'min_samples_leaf':min_leaf,
                              'max_depth':max_dep,
                              'cv_logloss':np.mean(errors_fold)}

                    results.append(result)
                    print("="*10 + " {}/{} ".format(len(results),nb_scenarios) + "="*10)
                    for key, value in result.items():
                        print("{} : {}".format(key,value))

    results = sorted(results, key = lambda x : x['cv_logloss'])
    best_result = results[0]

    with open('data/s2_meta/best_tuning_rf.json', 'w') as fp:
        json.dump(best_result, fp, indent=4)

    return results


def evaluate_model(model, X_val, y_val,plot=False):
    print("="*40)
    print("Model Evaluation")

    y_pred_proba = model.predict_proba(X_val)
    y_pred       = model.predict(X_val)
    cf = confusion_matrix(y_val,y_pred) 
    cf = cf.astype('float')
    for i in range(cf.shape[0]):
        sum_row = np.sum(cf[i,:])
        for j in range(cf.shape[1]):
            cf[i,j] = round(cf[i,j]/sum_row*100.0)
    
    # AUC
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba[:,1], pos_label=1)
    
    print(".. test log_loss  : {:0.2f} %".format(log_loss(y_val,y_pred_proba)*100))
    print(".. test AUC       : {:0.2f} %".format(auc(fpr, tpr)*100))
    print(".. confusion matrix (in %):")
    print(cf)
 
    return y_pred


def predict_on_test(mdl,scaler,cols_to_keep):
    
    full_test    = read_data('test')
    full_test    = full_test[[x for x in cols_to_keep if x not in ['logerror','transactiondate']]]
    full_test    = feature_engineering(full_test,isTest=True)
    
    tests        = np.array_split(full_test, 30)
    
    results = []
    for i, test in enumerate(tests):
        print("="*10+" Predictong on batch {}/{} ".format(i+1,len(tests)) + "="*10)
        X_test       = scaler.transform(test.values)
        y = mdl.predict_proba(X_test)
        bin_result = pd.DataFrame(columns=['parcelid','p_bad_fit','p_good_fit'])
        bin_result.parcelid = test.index.values
        bin_result['p_bad_fit'] = y[:,0]
        bin_result['p_good_fit'] = y[:,1] 
     
        results.append(bin_result)
        
        del y
        del X_test
        gc.collect()
        
    results = pd.concat(results)
    results.to_csv('data/s1_intermediate/output_s210.csv',index=False)
#    alert(1)
    

def main() :

    df           = read_data()

    cols_to_keep = ['parcelid'] + select_most_present_features(df,load=True)
    df           = df[cols_to_keep]

    df           = read_bins(df)
    df.drop('logerror',axis=1,inplace=True)

    df           = feature_engineering(df)

    ## create train and test set
    X_train, X_val, y_train, y_val, scaler = create_inputs_model(df,
                                                                 objectif_var='bin',
                                                                 test_size = 0.25)

    ## tune model
#    tunings = tune_model(X_train,y_train,5)

    ## train model
    mdl         = train_model(X_train,y_train)

    ## predict on val set and evaluate
    y_val_pred  = evaluate_model(mdl, X_val,y_val,plot=False)
    y_val_pred  = y_val_pred
#    alert(1)

   ##predict test
    predict_on_test(mdl,scaler,cols_to_keep)


if __name__ == '__main__':
    main()
