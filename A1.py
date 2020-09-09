# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:53:10 2020

@author: Konstantin
"""


import pandas as pd
import numpy as np;
import scipy as sc;
import statsmodels.api as sm;
import matplotlib.pyplot as plt;
from pylab import *


def A1(A,l,n):

    t=linspace(1,len(A),len(A));
    I=linspace(1,1,len(A));

    X=np.mat(array([[I],[t]])).transpose();
    B=inv(X.transpose().dot(X)).dot(X.transpose()).dot(A);
    trend=B[0,0]+B[0,1]*t;
    A.index = pd.Index(t)


    #plt.figure();
    #plt.plot(A)
    #plt.plot(trend+(0.96*np.var(A))**0.5)
    #plt.plot(trend-(0.96*np.var(A))**0.5)
    #plt.plot(trend);
    #plt.show();

    A_adjust=A-trend;
     
    
    
    
    mod = sm.tsa.arima.ARIMA(A_adjust, order=(10, 0, 10));
    res = mod.fit();

    t2=linspace(1,len(A)+n,len(A)+n);
    trend2=B[0,0]+B[0,1]*t2;
    
    f=trend2+res.predict(l-l,l+n-1);
    
    
    err_train = np.mean(100*((A[l-30:len(A):1] - f[l-30:len(A):1])/A[l-30:len(A):1]));
    #print("Error="+str(err_train));

    f2=A[l-20::1];
    f=f[l-20::1];
    return (f, f2);
    




###############################################
#mod = sm.tsa.statespace.SARIMAX(A_adjust, trend='n', order=(0,1,0), seasonal_order=(1,1,1,7))
#results = mod.fit()
#print(results.summary())