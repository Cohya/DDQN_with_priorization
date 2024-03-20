
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
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())  
            
class AnnDualing_Opt1(object):
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

        val_adv_fc = Dense(K + 1, activation=tf.identity)(x2)
        
        ## Q(s,a) = V(s) + A(s,a)  - 1/|A| * sum(A(s,a_i))
        val =  tf.expand_dims(val_adv_fc[:, 0], -1) 
        adv = val_adv_fc[:, 1:]
        mean_adv = tf.reduce_mean(adv, -1, keepdims=True)
        
        q_s_a = val + adv - mean_adv
        
        self.model = Model(inputs = x_inputs, outputs = q_s_a)
        self.trainable_params = self.model.trainable_weights

    def forward(self, x):
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())  
            
            
class AnnDualing_Opt2(object):
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

        val = Dense(1, activation=tf.identity)(x2)
        adv = Dense(K, activation = tf.identity)(x2)
        
        ## Q(s,a) = V(s) + A(s,a)  - 1/|A| * sum(A(s,a_i))

        mean_adv = tf.reduce_mean(adv, -1, keepdims=True)
        
        
        q_s_a = val + adv - mean_adv
        
        self.model = Model(inputs = x_inputs, outputs = q_s_a)
        self.trainable_params = self.model.trainable_weights

    def forward(self, x):
        return self.model.call(x)#self.model.apply(x)
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())  
            

