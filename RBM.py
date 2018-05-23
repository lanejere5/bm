# RBM class
'''
    A class defining a restricted Boltzmann machine.

'''
import numpy as np
import random
import matplotlib.pyplot as plt

learning_rate = 0.01

def sigmoid(x):
    return np.divide(1., 1.+np.exp(-x))

class RBM:

    kind = 'RBM'

    def __init__(self,v_dim,h_dim):
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.W = np.zeros((self.v_dim,self.h_dim))
        self.a = np.zeros((self.v_dim,1))
        self.b = np.zeros((self.h_dim,1))
        return

    def generate_v_sample(self,h):
        '''
            Input:
            - h has shape (h_dim,m)
            - a has shape (v_dim,m)
            - W has shape (v_dim,h_dim)
        '''
        #assert(np.shape(h)[0] == self.h_dim)
        v_probs = sigmoid(self.a + np.dot(self.W,h))
        assert(not np.sum(np.isnan(v_probs)))
        return np.random.binomial(1,v_probs)

    def generate_h_sample(self,v):
        '''
            Input:
            - v has shape (v_dim,m)
            - b has shape (h_dim,m)
            - W has shape (v_dim,h_dim)
        '''
        #assert(np.shape(v)[0] == self.v_dim)
        h_probs = sigmoid(self.b + np.dot(np.transpose(self.W),v))
        assert(not np.sum(np.isnan(h_probs)))
        return np.random.binomial(1,h_probs) # when n=1, binomial = bernoulli

    def gradient_update(self,v,learning_rate):
        '''
            Input:
            - v has shape (v_dim,m)
            - h has shape (h_dim,m)
            - a has shape (v_dim,m)
            - b has shape (h_dim,m)
            - W has shape (v_dim,h_dim)
        '''

        #assert(np.shape(v)[0] == self.v_dim)
        # sample distributions
        h       = self.generate_h_sample(v)
        v_prime = self.generate_v_sample(h)
        h_prime = self.generate_h_sample(v_prime)
        
        # compute the gradients, averaged over mini-batch
        DW = -np.mean(v[:,np.newaxis,:]*h[np.newaxis,:,:] - v_prime[:,np.newaxis,:]*h_prime[np.newaxis,:,:], axis = 2)
        Da = -np.mean(v - v_prime, axis = 1,keepdims = True).reshape((self.v_dim,1))
        Db = -np.mean(h - h_prime, axis = 1,keepdims = True).reshape((self.h_dim,1))
        
        # update weights and biases
        self.W = self.W + learning_rate*DW
        self.a = self.a + learning_rate*Da
        self.b = self.b + learning_rate*Db
        
        return

    def train(self, x, epochs = 10, batch_size = 16, learning_rate = learning_rate, plot = False):
        ''' 
            Input:
            - x has shape (v_dim, number_of_examples)

        '''
        #assert(np.shape(x)[0]==self.v_dim)
        # initialize weights and parameters
        self.W = np.random.normal(0.,0.01,size = (self.v_dim,self.h_dim))
        # visible bias a_i is initialized to ln(p_i/(1-p_i)), p_i = (proportion of examples where x_i = 1)
        self.a = (np.log(np.mean(x,axis = 1,keepdims=True)+1e-10) - np.log(1-np.mean(x,axis = 1,keepdims=True)+1e-10)) 
        self.b = np.zeros((self.h_dim,1))
        
        num_minibatches = int(np.shape(x)[0]/batch_size)
        
        for i in range(epochs):
            print("Epoch %i"%(i+1))
            np.random.shuffle(x)
            
            for j in range(num_minibatches):
                v = x[:,i:i+batch_size]
                self.gradient_update(v,learning_rate)

            if plot == True:
                self.plot_weights()
            
        
        return

    def gibbs_sampling(self, n=1, m=1):
        '''
            n - number of iterations of blocked Gibbs sampling
        '''
        v_probs = np.full((self.v_dim,m),0.5)
        v = np.random.binomial(1,v_probs)
        #print(np.shape(v))
        
        h = self.generate_h_sample(v)
        
        for i in range(n):
            v = self.generate_v_sample(h)
            h = self.generate_h_sample(v)
        
        return v,h

    def plot_weights(self):
        plt.figure(1)

        plt.subplot(311)
        plt.title('Weights')
        plt.hist(self.W.flatten(),bins='auto')
        
        plt.subplot(312)
        plt.title('Visible biases')
        plt.hist(self.a.flatten(),bins='auto')
        
        plt.subplot(313)
        plt.title('Hidden biases')
        plt.hist(self.b.flatten(),bins='auto')

        plt.tight_layout()

        plt.show()



