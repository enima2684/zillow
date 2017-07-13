# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:34:16 2017

@author: Bouamama Amine
create bins and save result
"""

import numpy as np

from utils import read_data, read_params


def create_bins(df,q1,q2):
    s1 = df.logerror.quantile(q1)
    s2 = df.logerror.quantile(q2)
    df['bin'] = 1
    df.loc[df.logerror <= s1,'bin'] = 0
    df.loc[df.logerror >= s2,'bin'] = 2
    return df

def calculate_logerror_per_bin(df,method=np.median):
    res = df.groupby('bin',as_index=False).logerror.aggregate(method)
    res.rename(columns={'logerror':'pred_logerror'},inplace=True)
    return res


def main():
    df     = read_data()
    params = read_params('s1')
    df     = create_bins(df,
                         params['q1'],
                         params['q2'])
    print(df.bin.value_counts())
    df.reset_index(inplace=True)
    
    # save the new bins
    df[['parcelid','bin']].to_csv('data/s1_intermediate/train_bins.csv',index=False)

    # calculate median output per bin and save it
    logerr_per_bin = calculate_logerror_per_bin(df)
    logerr_per_bin.to_csv('data/s1_intermediate/train_logerror_per_bin.csv',index=False)


if __name__ == '__main__':
    main()