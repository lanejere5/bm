# Autoencoder class
'''
    A class defining an autoencoder DFF neural network.

    The layers can be pretrained with deep boltzmann machines.

    After pre-training, the model is unrolled into a deep autoencoder.

    The model can then be tuned using minibatch gradient descent.

    After training, one can compute encodings using the encode function.

'''
import numpy as np
from RBM import *

from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K

learning_rate = 0.01

class Autoencoder:

    kind = 'Autoencoder'

    def __init__(self,v_dim,latent_dim,num_hidden_layers):
        '''
            Inputs:

            - v_dim      = dimension of the visible units (input data)
            - latent_dim = dimension of the encoded layer
            - num_hidden_layers = number of hidden layers in the encoder. 

            Note that the number of hidden layers in the autoencoder when training
            will be twice num_hidden_layers. 
        '''

        self.latent_dim = latent_dim
        self.v_dim = v_dim
        self.num_hidden_layers = num_hidden_layers
        self.layer_dims = np.linspace(self.v_dim,self.latent_dim, num = self.num_hidden_layers+1).astype(int)

        print("Layer dimensions:")
        for i in range(self.num_hidden_layers+1):
            print("Layer %i: %i"%(i,self.layer_dims[i]))

        self.W = []
        self.b = []
        self.a = []
        self.pretrained = False

        self.autoencoder = None
        self.encoder = None
        self.decoder = None

        return

    def pretrain(self,x,epochs,num_samples = 2):
        '''
            Greedy layer-wise training

            shape(x) = (v_dim, number_of_examples)
        '''

        RBM_layers = []

        for i in range(self.num_hidden_layers): # initialize RBM's
            RBM_layers.append(RBM(self.layer_dims[i],self.layer_dims[i+1]))
        
        for i in range(self.num_hidden_layers):  # train RBM's 
            print("Training layer %i"%(i+1))

            RBM_layers[i].train(x,epochs) # train the ith RBM
            
            if not(i == self.num_hidden_layers - 1): # generate samples to train next layer
                _,x = RBM_layers[i].gibbs_sampling(2,num_samples) 

            self.W.append(RBM_layers[i].W) # save trained weights
            self.b.append(RBM_layers[i].b)
            self.a.append(RBM_layers[i].a)

        self.pretrained = True

        return

    
    def unroll(self):
        '''
            Unrolls the pretrained RBM network into a DFF keras model 
            and sets hidden layer parameters to pretrained values.
        '''
        if self.pretrained == False:
            print("Model not pretrained.")
            return

        # define keras model structure
        inputs = Input(shape=(self.v_dim,))
        x = inputs

        # build encoder 
        for i in range(self.num_hidden_layers):
            weights = [self.W[i],self.b[i].flatten()]
            x = Dense(self.layer_dims[i+1], activation='sigmoid', weights = weights)(x)

        # build decoder
        for i in range(self.num_hidden_layers):
            weights = [self.W[self.num_hidden_layers-i-1].T,self.a[self.num_hidden_layers-i-1].flatten().T]
            x = Dense(self.layer_dims[self.num_hidden_layers-i-1], activation='sigmoid', weights = weights)(x)

        model = Model(inputs,x)

        self.autoencoder = model


        return


    def train(self,x,epochs,learning_rate = 0.001,batch_size = 16):
        '''
            Fine-tune the autoencoder with minibatch gradient descent
        '''
        self.autoencoder.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')
        self.autoencoder.fit(x.T, x.T, epochs = epochs, batch_size=batch_size)

        return

    def encode(self):
        # 

        return

    def generate(self):
        # generate examples from the latent space

        return





