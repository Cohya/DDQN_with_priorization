
import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np 
import pickle 
from NoisyLayer import NoisyDense2
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
            
            
# class AnnDualing_Opt2(object):
#     def __init__(self, input_shape, K, num_atoms):
#         """
#         dictionary_details = {'LASTM':[[nuerons, activation],...],
#                               'Dense': [[neurons, activation],....],
#                               }
#         K - is output neurons number (last layer)
#         """
#         self.num_atoms = num_atoms # M
#         self.support = tf.ones(shape = (num_atoms,1))
        
#         input_shape = (np.prod(input_shape),)
#         x_inputs = Input(shape = input_shape)
#         x1 = Dense(40, activation=tf.nn.relu)(x_inputs)
#         x2= Dense(40, activation=tf.nn.relu)(x1)

#         val = Dense(1 * num_atoms, activation=tf.identity)(x2) # M
#         adv = Dense(K * num_atoms, activation = tf.identity)(x2) # AxM
#         # N = adv.shape[0]
#         adv = tf.keras.layers.Reshape((K,num_atoms))(adv)
#         ## Q(s,a) = V(s) + A(s,a)  - 1/|A| * sum(A(s,a_i))
        
                
#         val = tf.keras.layers.Reshape((1,num_atoms))(val) #  tf.reshape(val, shape = (N,1,num_atoms)) 
        
#         val = tf.tile(val, [1,K,1])
        
        
#         mean_adv = tf.reduce_mean(adv, -1, keepdims=True)
        
        
#         # q_s_a_atoms = val, adv - mean_adv ## Ammmmm... I dont think we can do it like this 
        
#         q_s_a_atoms = tf.math.add(val, adv) - mean_adv 
        
        
#         distribution = tf.math.softmax(q_s_a_atoms, axis = -1)
        
        
#         ## Avoid nan
#         cliped_dist = tf.clip_by_value(distribution, 
#                                        clip_value_min = 1e-3,
#                                        clip_value_max = float('inf'))
       
        
#         q_s_a = tf.linalg.matmul(cliped_dist, self.support) 
#         q_s_a = tf.squeeze(q_s_a, axis = -1)
#         self.model = Model(inputs = x_inputs, outputs = (q_s_a, cliped_dist))
#         self.trainable_params = self.model.trainable_weights

#     def forward(self, x):
#         q_s_a, cliped_dist = self.model.call(x)
#         return  q_s_a, cliped_dist #self.model.apply(x)
    
    
#     def copy_params_from(self, params):
#         for i in range(len(self.trainable_params)):
#             self.trainable_params[i].assign(params[i].numpy())  
            
class AnnDualing_Opt_distribution(tf.keras.Model):
    def __init__(self, input_shape, K, num_atoms, support):
        """
        dictionary_details = {'LASTM':[[nuerons, activation],...],
                              'Dense': [[neurons, activation],....],
                              }
        K - is output neurons number (last layer)
        """
        super().__init__()
        self.num_atoms = num_atoms # M
         #np.array([[1],[2],[3]]).astype(np.float32) # tf.ones(shape = (num_atoms,1))
        self.K  = K
        # input_shape = (np.prod(input_shape),)
        # x_inputs = Input(shape = input_shape)
        self.d1 = NoisyDense2(40, activation=tf.nn.relu)#(x_inputs)
        self.d2 = NoisyDense2(40, activation=tf.nn.relu)#(x1)

        self.val_layers = NoisyDense2(int(1 * num_atoms), activation=tf.identity)#(x2) # M
        self.adv_layer = NoisyDense2(int(K * num_atoms), activation = tf.identity)#(x2) # AxM
        # N = adv.shape[0]
        self.reshape_adv = tf.keras.layers.Reshape((K,num_atoms))#(adv)
        ## Q(s,a) = V(s) + A(s,a)  - 1/|A| * sum(A(s,a_i))
        self.support = support # (num_atoms,1)
                
        self.reshape_val = tf.keras.layers.Reshape((1,num_atoms))#(val) #  tf.reshape(val, shape = (N,1,num_atoms)) 
        
        ## Initiate the net 
        x = tf.random.normal(shape = (3,input_shape))
        self.call(x)
        #####################
        
        self.trainable_params = self.trainable_variables
        
    def reset_noise(self):
        self.d1.reset_noise()
        self.d2.reset_noise()
        self.val_layers.reset_noise()
        self.adv_layer.reset_noise()
        
    def remove_noise(self):
        self.d1.remove_noise()
        self.d2.remove_noise()
        self.val_layers.remove_noise()
        self.adv_layer.remove_noise()
        print("removed noised")
        
    def call(self, x, log  = False, training = False):
        
        # input_shape = (np.prod(input_shape),)
        # x_inputs = Input(shape = input_shape)
        x1 = self.d1(x, training = training)#
        x2 = self.d2(x1, training = training) 

        val  = self.val_layers(x2,training = training) # M
        adv = self.adv_layer(x2,training = training) # AxM
        # N = adv.shape[0]
        adv = self.reshape_adv(adv) # (N, K,num_atoms)
        ## Q(s,a) = V(s) + A(s,a)  - 1/|A| * sum(A(s,a_i))
        
        val = self.reshape_val(val) #  tf.reshape(val, shape = (N,1,num_atoms)) 
        
        val = tf.tile(val, [1,self.K,1]) # (N, K, num_atoms)
        
        
        mean_adv = tf.reduce_mean(adv, -1, keepdims=True) #(N, K)
        
        
        # q_s_a_atoms = val, adv - mean_adv ## Ammmmm... I dont think we can do it like this 
        
        q_s_a_atoms = tf.math.add(val, adv) - mean_adv 
        
        if log:
            distribution = tf.math.log_softmax(q_s_a_atoms, axis = -1)
        else:
            distribution = tf.math.softmax(q_s_a_atoms, axis = -1)
            ## Avoid nan
            distribution = tf.clip_by_value(distribution, 
                                            clip_value_min = 1e-3,
                                            clip_value_max = float('inf'))
            
        

       
        
        q_s_a = tf.linalg.matmul(distribution, self.support) 
        
        q_s_a = tf.squeeze(q_s_a, axis = -1)
  
        return  q_s_a, distribution
    
    
    def copy_params_from(self, params):
        for i in range(len(self.trainable_params)):
            self.trainable_params[i].assign(params[i].numpy())  

# num_atoms = 3
# K = 2
# N = 4
# x = tf.random.normal(shape = (N,4))


# ann = AnnDualing_Opt2(input_shape = 4, K = K , num_atoms=num_atoms)
# ann2 = AnnDualing_Opt_distribution(input_shape = 4, K= K, num_atoms = num_atoms)
# q_s_a, cliped_dist  = ann2(x, log = True)

# q_s_a_2, cliped_dist_2 = ann2(x)
# x1_layer = Dense(40, activation=tf.nn.relu)
# x2_layer = Dense(40, activation=tf.nn.relu)

# val_layer = Dense(1 * num_atoms, activation=tf.identity) # M
# adv_layer = Dense(K * num_atoms, activation = tf.identity) # AxM

# ## Q(s,a) = V(s) + A(s,a)  - 1/|A| * sum(A(s,a_i))
# x1 = x1_layer(x)
# x2 = x2_layer(x1)
# val = val_layer(x2)
# adv = adv_layer(x2)
# adv = tf.reshape(adv, shape = (N , K,  num_atoms))

# print(adv[1,1,:])
# print(val[1,:])

# f = tf.reshape(val, shape = (N,1,num_atoms)) 
# f = tf.tile(f, [1,K,1])

# print(tf.math.add(f, adv))



# mean_adv = tf.reduce_mean(adv, -1, keepdims=True)


# q_s_a_atoms = tf.math.add(f, adv) - mean_adv ## Ammmmm... I dont think we can do it like this 

# distribution = tf.math.softmax(q_s_a_atoms, axis = -1)