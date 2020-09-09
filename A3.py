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


def A3(A,l,n,s):

    mod = sm.tsa.statespace.SARIMAX(A, trend='n', order=(0,1,0), seasonal_order=(12,1,12,s));
    res = mod.fit()
   # plt.figure();
    #plt.title('The predict price EUR/USD (D1)');
    #plt.plot(A[l-20::1],linewidth=0.9);
    #plt.grid();
    #plt.plot(res.predict(l-20,l+n),linewidth=0.9,label='Спрогнозированные значения');
    #legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large');
    #plt.show()

    
    err_train = np.mean(100*((A[l-30::1] - res.predict(l-30,l))/A[l-30::1]));

    #print("Error="+str(err_train));
    f=res.predict(l-20,l+n);
    f2=A[l-20::1];
    
    return (f, f2);
    




###############################################
#mod = sm.tsa.statespace.SARIMAX(A_adjust, trend='n', order=(0,1,0), seasonal_order=(1,1,1,7))
#results = mod.fit()
#print(results.summary())