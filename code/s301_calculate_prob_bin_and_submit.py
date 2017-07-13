# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:19:14 2017

@author: Bouamama Amine


"""

import pandas as pd
from datetime import datetime
import gc

from utils import alert
from utils import read_logerror_per_bin
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
                        submit['logerror_1'] * submit['p_good_fit']+
                        submit['logerror_2'] * submit['p_bad_fit'] * submit['p_over_est'] )
                        
    
    
    submit['201610'] = submit['pred']
    submit['201611'] = submit['pred']
    submit['201612'] = submit['pred']
    submit['201710'] = submit['pred']
    submit['201711'] = submit['pred']
    submit['201712'] = submit['pred']
    
    submit = submit[['parcelid','201610','201611','201612','201710','201711','201712']]
    
    submit.to_csv("submissions/submission"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv",index=False)
    alert(2)
    
    
if __name__ == '__main__':
    main()