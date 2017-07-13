# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:19:14 2017

@author: Bouamama Amine


"""

import pandas as pd
from datetime import datetime

from utils import read_data, alert
from utils import read_classification_output, read_logerror_per_bin

def main():
    
    submit = read_classification_output()
    
    logerror_per_bin = read_logerror_per_bin()
    logerror_per_bin.set_index('bin',inplace=True)
    logerror_per_bin = logerror_per_bin.to_dict()
       
    submit['pred'] = submit.apply(lambda r : logerror_per_bin['pred_logerror'][0] * r['0'] + 
                                             logerror_per_bin['pred_logerror'][1] * r['1'] + 
                                             logerror_per_bin['pred_logerror'][2] * r['2'],axis=1)
    
    submit['201610'] = submit['pred']
    submit['201611'] = submit['pred']
    submit['201612'] = submit['pred']
    submit['201710'] = submit['pred']
    submit['201711'] = submit['pred']
    submit['201712'] = submit['pred']
    
    submit.drop(['0','1','2','pred'],axis=1,inplace=True)
    
    submit.to_csv("submissions/submission"+datetime.now().strftime("%Y%m%d%H%M%S")+".csv",index=False)
    alert(3)
    
    
if __name__ == '__main__':
    main()