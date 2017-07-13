# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 19:24:53 2017

@author: Bouamama Amine

Predict the most important features :
- Predict the outcome
- Error between model and prediction
- Distribution
- Assign class depending on quantile
- Predict class with RF
- Most important features

1. feature 11 - yearbuilt :(0.238964)
2. feature 7 - longitude :(0.105322)
3. feature 6 - latitude :(0.089600)
4. feature 9 - propertylandusetypeid :(0.085127)
5. feature 13 - taxvaluedollarcnt :(0.082658)
6. feature 15 - landtaxvaluedollarcnt :(0.070659)
7. feature 12 - structuretaxvaluedollarcnt :(0.066345)
8. feature 16 - taxamount :(0.059037)
9. feature 3 - calculatedfinishedsquarefeet :(0.049549)
10. feature 8 - lotsizesquarefeet :(0.046439)
11. feature 4 - finishedsquarefeet12 :(0.037019)
12. feature 0 - bathroomcnt :(0.021042)
13. feature 1 - bedroomcnt :(0.016465)
14. feature 2 - calculatedbathnbr :(0.012378)
15. feature 5 - fullbathcnt :(0.010251)
16. feature 10 - roomcnt :(0.009146)
17. feature 14 - assessmentyear :(0.000000)

"""


import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble.forest import RandomForestRegressor, RandomForestClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

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
    
    # keep only good data
#    q25 = df.logerror.quantile(0.1)
#    q75 = df.logerror.quantile(0.9)
#    df = df[(df.logerror<q75) & (df.logerror > q25)]
    
    if verbose :
        print(df.dtypes)
        
    return df

def create_inputs_model(df,test_size=0.3):
    
    X,y = df.drop('logerror',axis=1).values, df['logerror'].values
    
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
    rf = RandomForestRegressor(n_estimators=500,max_depth=5,n_jobs=-1,verbose=2)
    rf.fit(X_train,y_train)
    y_pred_train = rf.predict(X_train)
    print(".. training RMSE : {:0.3f} %".format(mean_squared_error(y_train,y_pred_train)*100))
    #print(".. training R2   : {:0.3f} %".format(r2_score(y_train,y_pred_train)*100))
    print(".. training MAE  : {:0.3f} %".format(mean_absolute_error(y_train,y_pred_train)*100))
    return rf
    
def evaluate_model(model, X_val, y_val,plot=False):
    y_pred = model.predict(X_val)
    print(".. test RMSE : {:0.3f} %".format(mean_squared_error(y_val,y_pred)*100))
    #print(".. test R2   : {:0.3f} %".format(r2_score(y_val,y_pred)*100))
    print(".. test MAE  : {:0.3f} %".format(mean_absolute_error(y_val,y_pred)*100))
    if plot:
        plt.scatter(y_pred,y_val,s=1,alpha=0.7)
        plt.plot(y_pred,y_pred,alpha=0.7,c='r')
    return y_pred



""" ============= MAIN ============== """

# read data
df           = read_data()

# select only relevant features
cols_to_keep = select_most_present_features(df,load=True)
df           = df[cols_to_keep]


# feature engineering
df           = feature_engineering(df)


# create train and test set
X_train, X_val, y_train, y_val, scaler = create_inputs_model(df,test_size = 0.3)


# train model
mdl         = train_model(X_train,y_train)

# predict on val set and evaluate
y_val_pred  = evaluate_model(mdl, X_val,y_val,plot=True)

# create vector of errors
errors = pd.DataFrame(y_val_pred - y_val, columns=['error'])
q80 = errors.error.quantile(0.8)
q20 = errors.error.quantile(0.2)
errors['group'] = 0
errors.loc[errors.error < q20, 'group'] = 1
errors.loc[errors.error > q80, 'group'] = 1

# Predict the class
rfc = RandomForestClassifier(n_estimators=500,max_depth=7,verbose=2,n_jobs=-1)
rfc.fit(X_val,errors.group.values) 

#most important features
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

feat_names = df.drop('logerror',axis=1).columns.values  
for f in range(X_val.shape[1]):
    print("{}. feature {} - {} :({:.06f})".format(f+1,indices[f],feat_names[indices[f]],importances[indices[f]]))
    
catalog = describe_features(df)

df_val          = pd.DataFrame(X_val,columns=feat_names)
df_val['group'] = errors.group.values
df_val['error'] = errors.error.values


# explore bathroomcnt
col = "bathroomcnt"
sns.countplot(df[col])
sns.pointplot(x=col,y='logerror',data=df)


sns.pointplot(x=col,y='error',data=df_val)
