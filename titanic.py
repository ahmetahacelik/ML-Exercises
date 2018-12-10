#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ahmetahacelik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler




veriler = pd.read_csv('train.csv')
veriler2=pd.read_csv('test.csv')


x = veriler.iloc[:,[2,5,6,7]].values #bağımsız değişkenler
y = veriler.iloc[:,1:2].values #bağımlı değişken

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
y2 = veriler2.iloc[:,[1,4,5,6]].values 
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(y2)
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_prediction=gnb.predict(X_test)





    
    

