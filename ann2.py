'''
Author: J. Rafid Siddiqui
Azad-Academy
jrs@azaditech.com
https://www.azaditech.com

'''
#==================================================================

from utils import *
import random

class ANN2:
    
    inputs = None
    params = None
    outputs = None
    activations = None
    num_params = (5*3)+(5*6)+(1*6)
    num_layers = 3
    Y = np.array([[1]])
    Y_hat = np.array([[1]])
    grad = None
    
    def __init__(self): 
        self.inputs = np.array([[1,0.5,0.5]])
        self.params = np.random.rand(self.num_params)
        self.outputs = [None]*self.num_layers
        self.activations = [None]*self.num_layers
        self.grad = np.random.rand(self.num_params)
    
    def cost(self,nnparams,L):
    
        #W1,W2 = weights2matrices(nnparams)
        self.params = nnparams
        J = self.feed_forward_step(L)
        
        return J

    def Gradient(self,nnparams,L):

        #W1,W2 = weights2matrices(nnparams)
        self.params = nnparams
        W1_grad, W2_grad, W3_grad, delta = self.back_prop_step(L)
        
        #W1 = (W1-0.85*W1_grad).flatten()
        #W2 = (W2-0.85*W2_grad).flatten()
        
        #self.params = np.concatenate((W1,W2))
            
        G = np.concatenate((W1_grad.flatten(),W2_grad.flatten(),W3_grad.flatten()))

        self.grad = G
        
        return G

    def feed_forward_step(self, L=1):
        
        X = self.inputs
        Y = self.Y
        W1,W2,W3 = weights2matrices2(self.params)

        z1 =  X @ np.transpose(W1)
        a1 = sigmoid(z1)
        a11 = np.concatenate((np.ones((a1.shape[0],1)),a1),axis=1)

        z2 = a11 @ np.transpose(W2)
        a2 = sigmoid(z2)
        a22 = np.concatenate((np.ones((a2.shape[0],1)),a2),axis=1)

        z3 = a22 @ np.transpose(W3)
        a3 = sigmoid(z3)

        m = X.shape[0]

        Jreg = (L/(2*m))*(np.sum(np.sum(W1[:,2:]**2,axis=1)) + np.sum(np.sum(W2[:,2:]**2,axis=1)) + np.sum(np.sum(W3[:,2:]**2,axis=1)))
        J = (-1/m*np.sum(np.sum(Y*np.log(a3)+(1-Y)*np.log(1-a3),axis=1))) + Jreg
        
        self.outputs = [z1,z2,z3]
        self.activations = [a1,a2,a3]
        self.Y_hat = a3

        return J            
            

    def back_prop_step(self,L=1):

        X = self.inputs
        Y = self.Y
        W1,W2,W3 = weights2matrices2(self.params)

        a1 = self.activations[0]
        a2 = self.activations[1]
        a3 = self.activations[2]

        z1 = self.outputs[0]
        z2 = self.outputs[1]

        

        delta3 = a3 - Y
        delta2 = (delta3 @ W3[:,1:])*sigmoid_grad(z2)

        #print("z2:{}".format(z2.shape))
        #print("W2:{}".format(W2.shape))
        

        delta1 = (delta2 @ W2[:,1:])*sigmoid_grad(z1)
        
        Delta1 = np.zeros(W1.shape)
        Delta2 = np.zeros(W2.shape)
        Delta3 = np.zeros(W3.shape)
        
        a11 = np.concatenate((np.ones((a1.shape[0],1)),a1),axis=1)
        a22 = np.concatenate((np.ones((a2.shape[0],1)),a2),axis=1)
        

        m = X.shape[0]
        for i in range(0,m):
            
            Delta3 = Delta3 + delta3[i,:] * a22[i,:]
            Delta2 = Delta2 + np.transpose(delta2[i,:]).reshape(5,1) @ a11[i,:].reshape(1,6)
            Delta1 = Delta1 + np.transpose(delta1[i,:]).reshape(5,1) @ X[i,:].reshape(1,3)
        
        W1_grad = 1/m * (Delta1 + L*np.concatenate((np.zeros((W1.shape[0],1)),W1[:,1:]),axis=1))
        W2_grad = 1/m * (Delta2 + L*np.concatenate((np.zeros((W2.shape[0],1)),W2[:,1:]),axis=1))
        W3_grad = 1/m * (Delta3 + L*np.concatenate((np.zeros((W3.shape[0],1)),W3[:,1:]),axis=1))
        

        return W1_grad,W2_grad,W3_grad,[delta1,delta2]
    
    def predict(self,X):

        num_pts = X.shape[0]
        Y = np.zeros(num_pts)
        for i in range(0,num_pts):
        
            self.inputs = np.array([X[i,:]])
            self.feed_forward_step()
            Y[i] = self.activations[2]
            #set_values(self.inputs[0,1],self.inputs[0,2],self.Y[0,0],self.outputs[0],self.outputs[1],self.activations[0],self.activations[1])

        return Y

