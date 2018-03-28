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

def CostFunction():
  return None

def Predict(model, features, bestFit = True):
  print(len(features))
  model[0].Inputs = features
  model[0].SizeOfInputs = len(features)
  
  for i in range(1,len(model)):
    model[i-1].Activate()
    print(np.round(model[i-1].FunctionValue,2))
    print(model[i-1].FunctionValue.shape)
    model[i].Inputs = model[i-1].FunctionValue
    model[i].SizeOfInputs = len(model[i].Inputs)
  model[i].Activate() 
  if bestFit:
    print(np.round(model[len(model)-1].FunctionValue,2))
    print(model[len(model)-1].FunctionValue.shape)
    return np.argmax(model[len(model)-1].FunctionValue, axis=1)
  else:
    return model[len(model)-1].FunctionValue

def RegComponent(weight, m, regModifier = 0.01):
  return (regModifier/m)*(np.column_stack((np.zeros(len(weight)),weight[:,1:])))
  