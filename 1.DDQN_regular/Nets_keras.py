
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np 
import pickle 

class Lstm_dual(object):
    def __init__(self, input_shape, K):
        # K - is output neurons number (last layer)
        x_inputs = Input(shape=input_shape)
        x = LSTM(128, activation=tf.nn.tanh)(x_inputs)
        x= Dense(128, activation=tf.nn.relu)(x)
        x1= Dense(128, activation=tf.nn.relu)(x)
        
        value = Dense(1, activation = tf.identity)(x1)
        advantage = Dense(K, activation=tf.identity)(x1)
        
        y = value + (advantage - tf.reduce_mean(advantage))
        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights
        
    def forward(self, x):
        x = tf.squeeze(x, axis = 2)
        x = tf.transpose(x, (0,2,1))
        return self.model.call(x)#self.model.apply(x)
        



class AnnResNet(object):
    def __init__(self, input_shape, K):
        """
        dictionary_details = {'LASTM':[[nuerons, activation],...],
                              'Dense': [[neurons, activation],....],
                              }
        K - is output neurons number (last layer)
        """
        input_shape = (np.prod(input_shape),)
        x_inputs = Input(shape = input_shape)
        x1 = Dense(128, activation=tf.nn.relu)(x_inputs)
        x2= Dense(128, activation=tf.nn.relu)(x1)
        x3 = tf.add(x1, x2)
        x4= Dense(128, activation = tf.nn.relu)(x3)
        x5 = tf.add(x3,x4)
        y = Dense(K, activation=tf.identity)(x5)

        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights

    def forward(self, x):
        n,h,w,c = x.shape
        x = tf.reshape(x, shape = (n, h*w*c))
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())            
   
    
class Ann(object):
    def __init__(self, input_shape, K):
        """
        dictionary_details = {'LASTM':[[nuerons, activation],...],
                              'Dense': [[neurons, activation],....],
                              }
        K - is output neurons number (last layer)
        """
        input_shape = (np.prod(input_shape),)
        x_inputs = Input(shape = input_shape)
        x1 = Dense(40, activation=tf.nn.relu)(x_inputs)
        x2= Dense(40, activation=tf.nn.relu)(x1)
        y = Dense(K, activation=tf.identity)(x2)

        self.model = Model(inputs = x_inputs, outputs = y)
        self.trainable_params = self.model.trainable_weights

    def forward(self, x):
        n,h,w,c = x.shape
        x = tf.reshape(x, shape=(n,h,c,w))
        x = tf.transpose(x,perm = (0,2,1,3))
        x = tf.reshape(x, shape = (n, h*w*c))
        print()
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())  