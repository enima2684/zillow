# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:32:11 2017

@author: Bouamama Amine
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

np.random.seed(2017)

""" ====== LOGGING ========== """

"""dummy model (average)"""
#.. test RMSE : 2.679 %
#.. test MAE  : 6.895 %
#.. LB  : 0.0651281

"""dummy model (median)"""
#.. test RMSE : 2.682 %
#.. test MAE  : 6.863 %
#.. LB  :  0.0653607

"""one feature : yearbuilt - RF MAE"""
#.. test RMSE : 2.682 %
#.. test MAE  : 6.858 %
#.. LB  : 0.0653816

"""one feature : yearbuilt - spline interpolation"""
#.. LB  : 0.0651342


""" Insight """
# If we predict "perfectly" 3% of the data (extreme data), MAE goes to 4.808% ...
# If we predict "perfectly" 10% of the data (extreme data), MAE goes to 3.365 % ...
# If we predict "perfectly" 50% of the central data, MAE goes to 6.163 %
# The MAE is driven by extreme values -> these are the ones we should predict

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
    
    
    #↓ select only the data that came from RF
    """
    1. feature 11 - yearbuilt :(0.233000)
    2. feature 7 - longitude :(0.114982)
    3. feature 6 - latitude :(0.095289)
    4. feature 9 - propertylandusetypeid :(0.083342)
    5. feature 13 - taxvaluedollarcnt :(0.080010)
    6. feature 12 - structuretaxvaluedollarcnt :(0.063192)
    7. feature 15 - landtaxvaluedollarcnt :(0.058482)
    8. feature 16 - taxamount :(0.057315)
    9. feature 3 - calculatedfinishedsquarefeet :(0.053873)
    10. feature 8 - lotsizesquarefeet :(0.053396)
    11. feature 4 - finishedsquarefeet12 :(0.037005)
    12. feature 0 - bathroomcnt :(0.019275)
    13. feature 2 - calculatedbathnbr :(0.018602)
    14. feature 5 - fullbathcnt :(0.011716)
    15. feature 1 - bedroomcnt :(0.011483)
    16. feature 10 - roomcnt :(0.009041)
    17. feature 14 - assessmentyear :(0.000000)
    """
    
    features_to_keep = ['yearbuilt']
    
    if('logerror' in df.columns.values):
        features_to_keep.append('logerror')
    df = df[features_to_keep]
    
    
    # construct geographical feature
#    df['x'] = np.cos( df.latitude /180000000 * np.math.pi ) * np.cos( df.longitude /180000000 * np.math.pi  )
#    df['y'] = np.cos( df.latitude /180000000 * np.math.pi ) * np.sin( df.longitude /180000000 * np.math.pi )
#    df['z'] = np.sin( df.latitude /180000000 * np.math.pi )
#    
#    df['x2'] = df.x * df.x
#    df['y2'] = df.y * df.y
#    df['z2'] = df.z * df.z
        
    # fill nas with mean
    for ft in df.columns :
        df[ft] = df[ft].fillna(df[ft].mean())
    
    
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
#    rf = RandomForestRegressor(n_estimators=100,
#                               max_depth=3,
#                               n_jobs=-1,
#                               verbose=2,
#                               criterion='mae')
    rf = GradientBoostingRegressor(loss='lad',n_estimators=500)
    rf.fit(X_train,y_train)
    y_pred_train = rf.predict(X_train)
    print(".. training RMSE : {:0.3f} %".format(mean_squared_error(y_train,y_pred_train)*100))
    print(".. training MAE  : {:0.3f} %".format(mean_absolute_error(y_train,y_pred_train)*100))
    return rf

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


def evaluate_model(model, X_val, y_val,plot=False):
    print("="*40)
    print("Model Evaluation")
    
    y_pred = model.predict(X_val)
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

#
## feature engineering
df           = feature_engineering(df)


## create train and test set
X_train, X_val, y_train, y_val, scaler = create_inputs_model(df,test_size = 0.25)


## train model
#mdl         = train_model(X_train,y_train)
mdl         = dummy_trainer(X_train,y_train)

## predict on val set and evaluate
#y_val_pred  = evaluate_model(mdl, X_val,y_val,plot=False)
y_val_pred  = evaluate_dummy_model(mdl, X_val,y_val,y_train)


##predict test


## TEST generation
#test         = read_data('test')
#test         = test[[x for x in cols_to_keep if x not in ['logerror','transactiondate']]]
#test         = feature_engineering(test)
#X_test       = scaler.transform(test.values)
##y = mdl.predict(X_test)
#y = np.ones((X_test.shape[0],1)) * mdl
# submit
#submit = pd.DataFrame(test.index.values,columns=['ParcelId'])
#submit['201610'] = y
#submit['201611'] = y
#submit['201612'] = y
#submit['201710'] = y
#submit['201711'] = y
#submit['201712'] = y
#submit.to_csv("submissions/submission"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv",index=False)

##
#from scipy.interpolate import UnivariateSpline
#x = df.yearbuilt.values
#y = df.logerror.values
#s = UnivariateSpline(x, y,s=3)
#plt.scatter(x,s(x))
#y = s(test.yearbuilt.values)

#
#import seaborn as sns
#df['abs_error'] = np.abs(df.logerror)
#df['group'] = 'goodfit'
#df.loc[(df.logerror < df.logerror.quantile(0.25)),"group"] = 'sous-fit'
#df.loc[(df.logerror > df.logerror.quantile(0.75)),"group"] = 'sur-fit'
#
#sns.distplot(df[df.group=='goodfit'].yearbuilt,hist=False,label='goodfit')    
#sns.distplot(df[df.group=='sous-fit'].yearbuilt,hist=False,label='sous-fit')  
#sns.distplot(df[df.group=='sur-fit'].yearbuilt,hist=False,label='sur-fit')     
#       
#sns.violinplot(y='yearbuilt',data=df,x='group',split=True)
#bins = df.yearbuilt.quantile(np.linspace(0,1,20)).values
#bins = [df.yearbuilt.min(),1960,df.yearbuilt.max()]
#df['timebox'] = pd.cut(df.yearbuilt,bins)
#sns.violinplot(y='logerror',data=df,x='timebox',split=True)
#sns.pointplot(x='timebox',y='abs_error',estimator=np.median,data=df)

