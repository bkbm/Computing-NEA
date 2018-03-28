import numpy as np
import pandas
import NeuralNetworkModule as NN
import Helper as A

df = pandas.read_csv(r'zoo.csv')
inputarray = df.values
y = inputarray[:,17]
x = inputarray[:,1:]
animalNames = x[0,:]
print(df)
TrainingFeatures = x[:75,:]
TrainingLabels = y[:75]
TestingFeatures = x[75:,:]
TestingLabels = y[75:]
TrainingData = NN.Data(TrainingFeatures, TrainingLabels,7)
TrainingData.Results = TrainingData.FormatResults()
FirstLayer= NN.SigmoidNeuronLayer(TrainingData.Features,TrainingData.SizeOfData, A.InitialiseRandomWeights(TrainingData.NumberOfFeatures, 8))
FirstLayer.Activate()
OutputLayer = NN.SigmoidNeuronLayer(FirstLayer.ActivationValue, len(FirstLayer.ActivationValue), A.InitialiseRandomWeights(8,TrainingData.PossibleResults),True)
for ix in range(0,100):
  FirstLayer.Activate()
  OutputLayer.Activate()
  OutputLayer.CalculateError(TrainingData.Results)
  FirstLayer.CalculateError(outputDelta=OutputLayer.Delta,outputWeight = OutputLayer.Weight)
  OutputLayer.UpdateWeights(OutputLayer.SizeOfInputs, 0.3)
  FirstLayer.UpdateWeights(FirstLayer.SizeOfInputs,0.3)
  print('Iterate')
model = [FirstLayer.Weight, OutputLayer.Weight]
print(model[0])
