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
  