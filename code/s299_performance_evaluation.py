# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 00:30:14 2017

@author: Bouamama Amine

Performance evaluation
"""

import pandas as pd
from datetime import datetime
import gc
from sklearn.metrics import mean_absolute_error as mae

from utils import alert
from utils import read_logerror_per_bin, read_data
from utils import read_prob_bad_good_fit,read_prob_under_over_estimation

def main():
    
    bad_good_fit   = read_prob_bad_good_fit()
    over_under_est = read_prob_under_over_estimation()
    
    
    submit = pd.merge(
            left = bad_good_fit,
            right = over_under_est,
            on = 'parcelid',
            how = 'left')
    
    del bad_good_fit, over_under_est
    gc.collect()
    
    logerror_per_bin = read_logerror_per_bin()
    logerror_per_bin.set_index('bin',inplace=True)
    logerror_per_bin = logerror_per_bin.to_dict()['pred_logerror']
    
    submit['logerror_0'] = logerror_per_bin[0]
    submit['logerror_1'] = logerror_per_bin[1]
    submit['logerror_2'] = logerror_per_bin[2]
    
    submit['pred'] = (submit['logerror_0'] * submit['p_bad_fit'] * submit['p_under_est'] +
                      submit['logerror_1'] * submit['p_good_fit'] +
                      submit['logerror_2'] * submit['p_bad_fit'] * submit['p_over_est'] )
                        
    
    
    # filter only training data for which we have an output
    df = read_data()
    df = df[['parcelid','logerror']]
  
    df = df.merge(submit[['parcelid','pred']],
                  on='parcelid',
                  how='left')
    
    print('Final MAE : {:0.3f} %'.format(mae(df.logerror.values,df.pred.values)*100))
    
if __name__ == '__main__':
    main()