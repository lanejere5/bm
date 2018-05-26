# RBM class
'''
    A class defining a restricted Boltzmann machine.

'''
import numpy as np
import random
import matplotlib.pyplot as plt

learning_rate = 0.01

def sigmoid(x): 
    return 1/(1+np.exp(-x))

class RBM:

    kind = 'RBM'

    def __init__(self,v_dim,h_dim):
        '''
            v_dim = dimension of the visible layer
            h_dim = dimension of the hidden layer
        '''
        self.v_dim = v_dim
        self.h_dim = h_dim
        self.W = np.zeros((self.v_dim,self.h_dim))
        self.a = np.zeros((self.v_dim,1))
        self.b = np.zeros((self.h_dim,1))
        return

    @classmethod
    def from_Values(cls,weights):
        '''
            Initialize with trained weights.
        '''
        W,a,b = weights['W'],weights['a'],weights['b']
        assert (W.shape[0] == a.shape[0]) and (W.shape[1] == b.shape[0])
        rbm = cls(W.shape[0],W.shape[1])
        rbm.W = W
        rbm.a = a
        rbm.b = b
        return rbm

    @classmethod
    def from_File(cls,filename):
        '''
            Initialize with weights loaded from a file.
        '''
        return cls.from_Values(RBM.load_weights(filename))

    def generate_v_sample(self,h):
        '''
            Input:
            - h has shape (h_dim,m)
            - a has shape (v_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert(h.shape[0] == self.h_dim)
        v_probs = sigmoid(self.a + np.dot(self.W,h))
        assert(not np.sum(np.isnan(v_probs)))
        return np.random.binomial(1,v_probs)

    def generate_h_sample(self,v):
        '''
            Input:
            - v has shape (v_dim,m)
            - b has shape (h_dim,1)
            - W has shape (v_dim,h_dim)
        '''
        assert(v.shape[0] == self.v_dim)
        h_probs = sigmoid(self.b + np.dot(self.W.T,v))
        assert(not np.sum(np.isnan(h_probs)))
        return np.random.binomial(1,h_probs)

    def train(self, x, epochs = 10, batch_size = 20, learning_rate = learning_rate, plot = False, initialize_weights = True):
        ''' 
            Input:
            - x has shape (v_dim, number_of_examples)
            - if plot = True then RBM plots debugging related plots after every epoch
            - use initialize_weights = False if you are continuing to train a model (e.g loaded from earlier trained weights)

        '''
        assert(x.shape[0]==self.v_dim)

        np.random.seed(0)

        # hyperparameters used by Hinton et al. for MNIST
        initialmomentum  = 0.5
        finalmomentum    = 0.9
        weightcost       = 0.0002
        num_minibatches  = int(x.shape[1]/batch_size)

        DW = np.zeros((self.v_dim,self.h_dim))
        Da = np.zeros((self.v_dim,1))
        Db = np.zeros((self.h_dim,1))

        if initialize_weights == True: # initialize weights and parameters
            self.W = np.random.normal(0.,0.1,size = (self.v_dim,self.h_dim))
            # visible bias a_i is initialized to ln(p_i/(1-p_i)), p_i = (proportion of examples where x_i = 1)
            #self.a = (np.log(np.mean(x,axis = 1,keepdims=True)+1e-10) - np.log(1-np.mean(x,axis = 1,keepdims=True)+1e-10))
            self.a = np.zeros((self.v_dim,1))
            self.b = np.zeros((self.h_dim,1))

        for i in range(epochs):
            print("Epoch %i"%(i+1))
            np.random.shuffle(x.T)

            if i>5:
                momentum = finalmomentum
            else: 
                momentum = initialmomentum
            
            for j in range(num_minibatches):
                v = x[:,j*batch_size:(j+1)*batch_size]

                # sample distributions
                h     = self.generate_h_sample(v)
                v_neg = self.generate_v_sample(h)
                h_neg = self.generate_h_sample(v_neg)
                
                # compute the gradients, averaged over minibatch, with momentum and regularization
                DW = momentum*DW + learning_rate*(np.mean(v[:,np.newaxis,:]*h[np.newaxis,:,:] - v_neg[:,np.newaxis,:]*h_neg[np.newaxis,:,:], axis = 2)-weightcost*self.W)
                Da = momentum*Da + learning_rate*np.mean(v - v_neg, axis = 1,keepdims = True) #.reshape((self.v_dim,1))
                Db = momentum*Db + learning_rate*np.mean(h - h_neg, axis = 1,keepdims = True) #.reshape((self.h_dim,1))
                
                # update weights and biases
                self.W = self.W + DW
                self.a = self.a + Da
                self.b = self.b + Db

            if plot == True:
                self.plot_weight_histogram()
                self.plot_weights()

                v,_ = self.gibbs_sampling(1,1)
                plt.imshow(v.reshape((28,28)),cmap=plt.cm.gray)
                plt.show()

                v,_ = self.gibbs_sampling(1,1,x[:,0].reshape((self.v_dim,1)))
                plt.imshow(v.reshape((28,28)),cmap=plt.cm.gray)
                plt.show()

        return

    def gibbs_sampling(self, n=1, m=1,v=None):
        '''
            n - number of iterations of blocked Gibbs sampling
        '''
        if v is None:
            v_probs = np.full((self.v_dim,m),0.5)
            v = np.random.binomial(1,v_probs)

        h = self.generate_h_sample(v)
        for i in range(n):
            v = self.generate_v_sample(h)
            h = self.generate_h_sample(v)
        return v,h

    def plot_weights(self):
        '''
            For debugging 
        '''
        
        return

    def plot_weight_histogram(self):
        '''
            For debugging 
        '''
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
        return

    def save(self, filename):
        '''
            Save trained weights of self to file
        '''
        weights = {"W":self.W,"a":self.a,"b":self.b}
        RBM.save_weights(weights,filename)
        return

    @staticmethod
    def save_weights(weights,filename):
        '''
            Save RBM weights to file
        '''
        np.savetxt(filename + '_a.csv',weights['a'],delimiter=",")
        np.savetxt(filename + '_b.csv',weights['b'],delimiter=",")
        np.savetxt(filename + '_W.csv',weights['W'],delimiter=",")
        return

    @staticmethod
    def load_weights(filename):
        '''
            Save RBM weights to file
        '''
        W = np.loadtxt(filename + '_W.csv',delimiter=",")
        a = np.loadtxt(filename + '_a.csv',delimiter=",").reshape((W.shape[0],1))
        b = np.loadtxt(filename + '_b.csv',delimiter=",").reshape((W.shape[1],1))
        return {"W":W,"a":a,"b":b}


