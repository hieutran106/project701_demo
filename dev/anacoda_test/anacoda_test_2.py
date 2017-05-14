# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:46:49 2017

@author: hieutran
"""
import pandas as pd
df = pd.read_csv('colorFeatures.csv')
test=[0,1,4]
for i in test:    
    out_put = df.ix[i]
    print(out_put['file_name'])