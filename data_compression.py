# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:00:52 2023

@author: EVHARLEY
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest


def decrease_dimensionality():
    
    df = pd.read_pickle('ctfidf_vector.pkl')
    
    reduced = SelectKBest(k=10000).fit_transform(df.loc[:, df.columns != 0], 
                                                df.loc[:, df.columns == 0])
    reduced_df = pd.DataFrame(reduced)
    reduced_df['title'] = df[0]
    return reduced_df


if __name__ == '__main__':
    df = decrease_dimensionality()
    df.to_pickle('reduced.pkl')
    