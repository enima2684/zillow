# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 14:54:25 2017

@author: Bouamama Amine

POC :
    idea here is that what is important is to know what are the points we do not predict well.
    We create groups (under-estimation, over-estimation, good-fit) and just redict the median in each group
"""


import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import pickle

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
    

def create_bins(df,Q1,Q2):
    quantiles = [0] + [Q1,Q2] + [1]
    bins = df.logerror.quantile(quantiles).values
    group_names = [-1,0,1]
    df['bin'] = pd.cut(df.logerror,bins,labels=group_names)
    return df                         


def cross_val(df,K,q1=-1,q2=-1):
    print("="*15 + " START CV {:0.1f}% - {:0.1f}% ".format(q1*100,q2*100) + "="*15)
    cv_errors = []
    for k in range(K):
        # create train and val sets
        msk = np.random.rand(len(df)) < 0.7
        train = df[msk]
        val = df[~msk]
        
        # train the model
        model = train.groupby('bin',as_index=False).logerror.median()

        ## predict in vakidaiton test
        val = pd.merge(left=val,
                       right=model.rename(columns={'logerror':'pred_logerror'}),
                       on = 'bin',
                       how='left'
                       )
        ## evaluate
        error = mean_absolute_error(val.logerror.values,val.pred_logerror.values)
        print(".. test MAE in round {} : {:0.3f} %".format(k,error*100))

        cv_errors.append(error)
        
        
    print("="*5 + " FINAL RESULTS " + "="*5 + " :")
    print("CV score of {:0.3f}% +/- {:0.3f}%".format(np.mean(cv_errors)*100,np.std(cv_errors)*100))
    return cv_errors


""" ========= MAIN ========== """

## create bins
results = []
for q1 in np.linspace(0.005,0.4,80):
    print(q1)
    for q2 in np.linspace(0.6,0.995,80):
        df = read_data()
        df = create_bins(df,q1,q2)
        df = df[['logerror','bin']]
        df.dropna(inplace=True)
        errors = cross_val(df,10,q1,q2)
        result = {'errors' : errors,
                  'mean_mae' : np.mean(errors),
                  'q1' : q1,
                  'q2' : q2}
        results.append(result)

# save results
with open('data/s2_meta/cv_thresold_results.pkl', 'wb') as output:
    pickle.dump(results, output)


# POST PROCESSING
matrix_res = np.zeros((80,80))
min_error, q1_min, q2_min = 9999,-1,-1
for res in results :
    q1 = res['q1']
    q2 = res['q2']
    error = res['mean_mae']
    
    if(error < min_error):
        min_error = error
        q1_min = q1
        q2_min = q2
    
    delta1 = (0.4-0.005)/(80-1)
    delta2 = (0.995-0.6)/(80-1)
    i = round((q1 - 0.005)/delta1)
    j = round((q2 - 0.6)/delta2)
    
    matrix_res[i,j] = error
    
            
file = open("data/s2_meta/log_quantile_threshold.txt","w") 
file.write("="*15 + "FINAL THRESHOLDS "+"="*15)
file.write("\n..q1 : {:0.2f} %".format(q1_min*100))
file.write("\n..q2 : {:0.2f} %".format(q2_min*100))
file.write("\n..error : {:0.4f} %".format(min_error*100))
file.close()      
              

plt.imshow(matrix_res, cmap='hot', interpolation='nearest')
plt.show()

