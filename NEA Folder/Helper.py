import numpy as np
import NeuralNetworkModule as NN
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
            
def Softmax(z):
  scores =  np.exp(z)
  return scores/np.sum(scores, axis=1, keepdims=True)

def InitialiseRandomWeights(L_In,L_Out,N=None):
  if N != None:
            np.random.seed(N)
  epsilon = 0.12
  return (np.random.rand(L_Out,L_In+1)*2*epsilon)-epsilon

def CostFunction(count):
  print(count)

def Predict(model, features, bestFit = True):
  model[0].Inputs = features
  model[0].SizeOfInputs = len(features)
  
  for i in range(1,len(model)):
    model[i-1].Activate()
    model[i].Inputs = model[i-1].FunctionValue
    model[i].SizeOfInputs = len(model[i].Inputs)
    
  model[i].Activate() 
  
  if bestFit:
    return np.argmax(model[len(model)-1].FunctionValue, axis=1)
  else:
    return model[len(model)-1].FunctionValue

def RegComponent(weight, m, regModifier):
  return (regModifier/m)*(np.column_stack((np.zeros(len(weight)),weight[:,1:])))

def ClassificationAccuracy(model, features, results):
   prediction = Predict(model, features)
   count = 0
   for i in range(0,len(prediction)):
       print('Prediction: ', prediction[i])
       print('Result:', results[i])
       print(' ')
       if prediction[i] == results[i]:
            count += 1    
   return count/len(prediction)