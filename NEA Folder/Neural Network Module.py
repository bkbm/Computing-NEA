# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:42:30 2018

@author: baile
"""

import numpy as np
import Activations as A

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
class NeuronLayer:
    IsFinalLayer = None
    Weight = None
    ActivationValue = None
    RegComponent = None
    Error = None
    def __init__(self,weight,isFinalLayer = False):
        self.Weight = weight
        self.IsFinalLayer = isFinalLayer
                
    def Activate(self, features, sizeOfFeatures):
        features = np.column_stack((np.ones(sizeOfFeatures), features))
        self.ActivationValue = np.dot(self.Weight.T,features)
        
    def CalculateError(self,results, outputDelta, outputWeight):
        if self.IsFinalLayer:
            Delta = self.ActivationValue - results
            self.Error = np.dot(Delta.T,self.ActivationValue)
        else:
            Delta = np.dot(outputDelta,outputWeight)
            self.Error(np.dot(Delta.T,self.ActivationValue))
            
    def UpdateWeights(self,sizeOfdata,learningRate):
        weightGradient = (self.Error/sizeOfdata) + self.RegComponent
        self.Weight += -learningRate*weightGradient

class SigmoidNeuronLayer(NeuronLayer):
    SigmoidActivationValue = None
    def Activate(self, features, sizeOfFeatures):
        super().Activate(features,sizeOfFeatures)
        self.SigmoidActivationValue = 
    
            
    
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
 

