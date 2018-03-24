# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:42:30 2018

@author: baile
"""

import numpy as np

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

class NeuralNetwork:
    TrainingData = None
    SizeOfInput = None
    HiddenLayerSize = None
    Model = None
    
    def __init__(self,trainingData,hiddenlayerSize,model = None):
        self.TrainingData = trainingData
        self.SizeOfInput = trainingData.NumberOfFeatures
        self.HiddenLayerSize = hiddenlayerSize
        self.SizeOfOutput = trainingData.PossibleResults
        self.Model = model
    @classmethod
    def initWithoutDataObject(cls,trainingFeatures,trainingResults,possibleResults,hiddenLayerSize,model = {}):
        trainingData = Data(trainingFeatures,trainingResults,possibleResults)
        return cls(trainingData,hiddenLayerSize,model)
    def Sigmoid(self,z, derivative):
        if derivative:
            return self.Sigmoid(z, False)*(1-self.Sigmoid(z,False))
        else:
            return 1/(1+np.exp(-z)) 
    def Tanh(self,z,derivative=False):
        if derivative:
            return 1-self.Tanh(z)
        else :
            return np.tanh(z)
