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
FirstLayer= NN.SigmoidNeuronLayer(TrainingData.Features, A.InitialiseRandomWeights(TrainingData.NumberOfFeatures, 8, 0))
FirstLayer.Activate()
OutputLayer = NN.SigmoidNeuronLayer(FirstLayer.FunctionValue, A.InitialiseRandomWeights(8,TrainingData.PossibleResults, 0), isFinalLayer=True)
for ix in range(0,20000):
  FirstLayer.Activate()
  OutputLayer.Activate()
  OutputLayer.CalculateError(TrainingData.Results)
  FirstLayer.CalculateError(outputDelta=OutputLayer.Delta,outputWeight = OutputLayer.Weight)
  OutputLayer.UpdateWeights(OutputLayer.SizeOfInputs, 0.3)
  FirstLayer.UpdateWeights(FirstLayer.SizeOfInputs,0.3)
  if ix % 1000 == 0:
      print(A.CostFunction(ix))
model = [FirstLayer, OutputLayer]
FirstLayer.Inputs = TestingFeatures
FirstLayer.SizeOfInputs = len(TestingFeatures)
FirstLayer.Activate()
OutputLayer.Inputs = FirstLayer.FunctionValue
OutputLayer.SizeOfInputs = len(OutputLayer.Inputs)
OutputLayer.Activate()
predict = np.argmax(OutputLayer.FunctionValue,axis=1)
print(predict)
count = 0
for ix in range(0,len(predict)):
    if predict[ix]==TestingLabels[ix]:
        count += 1
print(count/len(predict))

