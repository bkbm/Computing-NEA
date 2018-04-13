# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 20:22:56 2018

@author: baile
"""

import pandas 
import numpy as np
import Helper
import AbstractClassNN as NN

def Normalise(value,mean,std):
    return((value-mean)/std)

def UnNormalise(value,mean,std):
    return (value*std)+mean
data = pandas.read_csv(r'bitcoin_price.csv')
print(data)
inputarray = data.values
price = inputarray[:,4]
print(price)
prices = [price[i-100:i] for i in range(100,len(price))]
prices2 = np.asarray(prices)
print(prices2)
print(prices2.shape)
#np.savetxt(r"foo.csv",prices2 , delimiter=",")
a=round(len(prices2) * 0.75)
trainingData = prices2[:a,:]
mean = np.mean(trainingData)
std = np.std(trainingData)
trainingData = Normalise(trainingData,mean,std)
testingData = prices2[a:,:]
tstMean = np.mean(testingData)
tstSTD = np.std(testingData)
Normalise(testingData,tstMean,tstSTD)
trainingfeatures = trainingData[:,:99]
trainingResult = trainingData[:,99]
testingFeatures = testingData[:,:99]
testingResults = testingData[:,99]
FeatureNo = 99
OutputNo = 1
hid1 = 50
hid2 = 25
trainingResult = trainingResult.reshape(1245,1)
print(trainingResult.shape)
W1 = Helper.InitialiseRandomWeights(99,50)
#W2 = Helper.InitialiseRandomWeights(50,25)
W3 = Helper.InitialiseRandomWeights(50,1)
firstLayer = NN.SigmoidNeuronLayer(W1,trainingfeatures)
#secondlayer = NN.SigmoidNeuronLayer(W2)
outputLayer = NN.LinearNeuronLayer(W3,finalLayer=True)
firstLayer.Activate()
outputLayer.Inputs = firstLayer.FunctionValue
outputLayer.Activate()
print(outputLayer.ActivationValue.shape)
temp = outputLayer.ActivationValue - trainingResult
print(temp.shape)
num_passes = 20000 
for i in range(0,num_passes):
    print(i)
    firstLayer.Activate()
#    secondlayer.Inputs = firstLayer.FunctionValue
#    secondlayer.Activate()
    outputLayer.Inputs = firstLayer.FunctionValue
    outputLayer.Activate()
    
    outputLayer.CalculateError(trainingResult)
#    secondlayer.CalculateError(outputDelta=outputLayer.Delta,outputWeight = outputLayer.Weight)
    firstLayer.CalculateError(outputDelta=outputLayer.Delta,outputWeight=outputLayer.Weight)
    outputLayer.UpdateWeights()
    firstLayer.UpdateWeights()
W1 = firstLayer.Weight
W3 = outputLayer.Weight
testFL = NN.SigmoidNeuronLayer(firstLayer.Weight,testingFeatures)
testOL =NN.LinearNeuronLayer(outputLayer.Weight)
testFL.Activate()
testOL.Inputs = testFL.FunctionValue
testOL.Activate()
prediction = testOL.ActivationValue
print('NormalisedPrediction: {} '.format(prediction))    
aPrediction = UnNormalise(prediction,tstMean,tstSTD)
Utest = UnNormalise(testingResults,tstMean,tstSTD)
[print('Unnormed Pred: {} {}'.format(aPrediction[i],Utest[i])) for i in range(0, len(aPrediction))]
print('done') 

