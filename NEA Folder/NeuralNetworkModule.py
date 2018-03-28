# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 11:42:30 2018

@author: baile
"""

import numpy as np
import Helper as A

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
	Inputs = None
	SizeOfInputs = None
	IsFinalLayer = None
	Weight = None
	ActivationValue = None
	RegModifier = None
	Delta = None
	Error = None

	def __init__(self, inputs, weight, regModifier,isFinalLayer=False):
		self.Inputs = inputs
		self.SizeOfInputs = len(inputs)
		self.Weight = weight
		self.RegModifier = regModifier
		self.IsFinalLayer = isFinalLayer

	def Activate(self):
	  
		features = np.column_stack((np.ones(self.SizeOfInputs), self.Inputs))
		self.ActivationValue = np.array(
		    np.dot(features, self.Weight.T), dtype=np.float32)

	def CalculateError(self, results=None, outputDelta=None,
	                   outputWeight=None):
		features = np.column_stack((np.ones(len(self.Inputs)), self.Inputs))
		if self.IsFinalLayer:
			self.Delta = self.ActivationValue - results
			self.Error = np.dot(self.Delta.T, features)
		else:
			self.Delta = np.dot(outputDelta, outputWeight)
			self.Error(np.dot(self.Delta.T, features))

	def UpdateWeights(self, sizeOfdata, learningRate):
		weightGradient = (self.Error / sizeOfdata) + A.RegComponent()
		Modifier = np.array((-learningRate * weightGradient), dtype=np.float32)
		self.Weight += Modifier


class SigmoidNeuronLayer(NeuronLayer):
	FunctionValue = None

	def Activate(self):
		super().Activate()
		self.FunctionValue = A.Sigmoid(self.ActivationValue)

	def CalculateError(self, results=None, outputDelta=None,
	                   outputWeight=None):
		features = np.column_stack((np.ones(len(self.Inputs)), self.Inputs))
		if self.IsFinalLayer:
			self.Delta = self.FunctionValue - results
			self.Error = np.dot(self.Delta.T, features)
		else:
			error = np.dot(outputDelta, outputWeight)
			self.Delta = error * A.Sigmoid(
			    np.column_stack((np.ones(len(self.ActivationValue)),
			                     self.ActivationValue)), True)
			self.Error = np.dot(self.Delta[:, 1:].T, features)


class TanHNeuronLayer(NeuronLayer):
	FunctionValue = None
	def Activate(self):
		super().Activate()
		self.FunctionnValue = A.Tanh(self.ActivationValue)
	
	def CalculateError(self, results=None, outputDelta=None, outputWeight=None):
	  features = np.column_stack((np.ones(len(self.Inputs)),self.Inputs))
	  if self.IsFinalLayer:
	    self.Delta = self.FunctionValue - results
	    self.Error = np.dot(self.Delta.T, features)
	  else:
	    error = np.dot(outputDelta, outputWeight)
	    self.Delta = error*A.Tanh(np.column_stack((np.ones(len(self.ActivationValue)),self.ActivationValue)), True)
	    self.Error = np.dot(self.Delta[:,1:].T,features)
	
class SoftmaxNeuronLayer(NeuronLayer):
  FunctionValue = None
  
  def __init__(self, inputs, sizeOfInputs, weight):
    super().__init__(inputs, sizeOfInputs, weight, True)
  
  def Activate(self):
    super().Activate()
    self.FunctionValue = A.Softmax(self.ActivationValue)
    
  def CalculateError(self, results):
    features = np.column_stack((np.ones(len(self.Inputs)),self.Inputs))
    self.Delta = self.SoftFunctionValue - results
    self.Error = np.dot(self.Delta.T, features)
    
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
 

