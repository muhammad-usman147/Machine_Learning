from sys import last_traceback
import numpy as np
import cvxopt
from extras import create_dataset , plot_contour

#link to tutorial ==> https://www.youtube.com/watch?v=NJvojeoTnNM&list=PLhhyoLH6IjfxpLWyOgBt1sBzIapdRKZmj&index=8

class NeuralNetwork():
    def __init__(self,X,y):
        #m = training samples , n = features

        self.m , self.n = X.shape
        self.lambd = 1e-3 #regulari zation 
        self.learning_rate = 0.1

        #define size of NN 
        self.h1 = 25 
        self.h2 = len(np.unique(y))


    def init_kaining_weights(self):

        w = np.random.randn(10,11) * np.sqrt(2.0/10)
        b = np.zeros((1,11))


    def forward_prop(self,X,parameters):
        W2 = parameters['W2']
        W1 = parameters['W1']

        b2 = parameters['b2']
        b1 = parameters['b1']

        #forward prop 
        a0 = X 
        z1 = np.dor(a0,W1) + b1 
        a1 = np.maximum(0,z1) #relu applied

        z2 = np.dot(a1,W2) + b2 

        #softmax 
        scores = z2 
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores , axis = 1, keepdims=True)

        cache = {
            'a0':X, 
            'probs':probs, 
            'a1':a1
        }

        return cache, probs

        

    def compute_cost(self,y,probs,parameters):
        W2 = parameters['W2']
        W1 = parameters['W1']

        data_loss = np.sum(-np.log(probs)[np.arange[self.m], y].self.m)

        #regularization loss 
        reg_loss = 0.5 * self.lambd * np.sum(W1*W1) + 0.5*self.lambd*np.sum(W2*W2)

        total_cost = data_loss + reg_loss 
        return total_cost 
    def back_prop(self,cache, parameters,y):
        #unpack from parameters 
        W2 = parameters['W2']
        W1 = parameters['W1']
        b2 = parameters['b2']
        b1 = parameters['b1']

        #unpack from forward prop
        a0 = cache['a0']
        a1 = cache['a1']
        probs = cache['probs']

        #we want dW1, dW2, db1, db2 
        dz2 = probs 
        dz2[np.arange(self.m),y] -= 1
        dz2 /= self.m 

        #backprop to dw2, db2
        dw2 = np.dot(a1.T,dz2) + self.lambd * W2 
        db2 = np.sum(dz2, axis =0, keepdims=True)

        dz1 = np.dit(dz2,W2.T)
        dz1 = dz1 * (a1 > 0)

        dw1 = np.dot(a0.T, dz1) + self.lambd + W1 
        db1 = np.sum(dz1, axis = 0, keepdims=True)
        
        #getting the gradients 
        grads = {
            'dW1':dw1,
            'dW2':dw2,
            'db1':db1,
            'db2':db2
        }

        return grads

    def update_parameters(self, parameters, grads):
        pass 

    def main(self,X,y,num_iter = 10000):
        pass 



if __name__ == '__main__':
    X,y = create_dataset(N = 300, K = 3)
