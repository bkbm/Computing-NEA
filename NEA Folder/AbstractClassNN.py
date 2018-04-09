# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:19:10 2018

@author: baile
"""

import numpy as np
import Helper as A
from abc import ABC, abstractmethod

class Data:
    Features = None
    Results = None
    NumberOfFeatures = None
    PossibleResults = None
    SizeOfData = None
    def __init__(self,features,results,possibleResults):       
        if len(features) == len(results):
            self.Features = features
            self.Results = results
        else:
            raise NameError('The number of rows in features needs to match that of results')
        if len(features.shape) == 1:
            self.NumberOfFeatures = 1
        else:
            self.NumberOfFeatures = features.shape[1]
        self.PossibleResults = possibleResults
        self.SizeOfData = len(features)
    def FormatResults(self):
        Categories = np.eye(self.PossibleResults)
        FormattedResults = np.zeros((self.SizeOfData,self.PossibleResults))
        for ix in range(0, self.SizeOfData):
            FormattedResults[ix,:] = Categories[self.Results[ix]-1,:]
        return FormattedResults
    
class NeuronLayer(ABC):
    @property 
    def Inputs(self):
        return self._Inputs
    @Inputs.setter
    def Inputs(self, value):
        self._Inputs = value
        if self._Inputs != None:
            self.SizeOfData = len(value)
    
    def __init__(self, weight, inputs = None, regModifier = 0.01, finalLayer=False):
        self.Inputs = inputs
        self.Weight = weight
        self.RegModifier = regModifier
        self.FinalLayer = finalLayer
        super().__init__()
               
    def Activate(self):
        features = A.BiasedInput(self.Inputs)
        self.ActivationValue = np.array(np.dot(features, self.Weight.T),dtype=np.float32)
    @abstractmethod
    def CalculateError(self):
        pass
    @abstractmethod
    def UpdateWeights(self):
        pass
    
class LinearNeuronLayer(NeuronLayer):
    def CalculateError(self, results = None, outputDelta = None, outputWeight = None):
        if self.FinalLayer:
            self.Delta = self.ActivationValue - results
        else:
            self.Delta = np.dot(outputDelta, outputWeight)
        self.Error = np.dot(self.Delta.T,A.BiasedInput(self.Inputs))
    
    def UpdateWeights(self, sizeOfdata, learningRate):
        weightGradient = (self.Error/sizeOfData)+ A.RegComponent(self.Weight,sizeOfdata,self.RegModifier)
        modifier = np.array((-learningRate*weightGradient), dtype=np.float32)
        self.Weight +=  modifier
    
class SigmoidNeuronLayer(NeuronLayer):
    
    def Activate(self):
        super().Activate()
        self.FunctionValue = A.Sigmoid(self.ActivationValue) 
    
    def CalculateError(self, results = None, outputDelta = None, outputWeight = None):
        if self.FinalLayer:
            self.Delta = self.FunctionValue - results
            self.Error = np.dot(self.Delta.T,A.BiasedInput(self.Inputs))
        else:
            self.Delta = np.dot(outputDelta,outputWeight)*A.Sigmoid(A.BiasedInput(self.ActivationValue), True)
            self.Error = np.dot(self.Delta[:,1:].T, A.BiasedInput(self.Inputs))
    def UpdateWeights(self,sizeOfData, learningRate):
        weightGradient = (self.Error/sizeOfData) + A.RegComponent(self.Weight,sizeOfData,self.RegModifier)
        modifier = np.array((-learningRate*weightGradient), dtype=np.float32)
        self.Weight += modifier
class TanhLayer(NeuronLayer):
    pass
class SoftMaxLayer(NeuronLayer):
    pass


        
        
        

    
    