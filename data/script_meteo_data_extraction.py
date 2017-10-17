# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:53:40 2017

@author: lucas
"""


TRAIN_NUMBER = 36

import pandas as pd
import numpy as np

# Concatenate all dataframes in only one
frames = []
for i in range(1, TRAIN_NUMBER+1):
    df = pd.read_csv('train_' + str(i) + '.csv', sep=";")
    frames.append(df)
data = pd.concat(frames)

# Replace all nAN values by 0
df = data.replace(np.nan, 0)

# Write file in same directory than script
data.to_csv('train_global.csv')