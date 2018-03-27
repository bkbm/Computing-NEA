# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:10:17 2018

@author: baile
"""
import numpy as np
def Sigmoid(z, derivative=False):
        if derivative:
            return Sigmoid(z, False)*(1-Sigmoid(z,False))
        else:
            return 1/(1+np.exp(-z)) 
        
def Tanh(z,derivative=False):
        if derivative:
            return 1-Tanh(z)
        else :
            return np.tanh(z)  