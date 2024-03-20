import numpy as np
import random  
import tensorflow as tf 
import os 
import pickle 
import time 

            
        
class DQN:
    def __init__(self, net, K, learning_rate = 0.0001, scope = 'model'):
        
        # # num_channels <-- the depth of the input 
        # self.dims_input_x_y = dims_input_x_y # the size of the x,y dimention of the input  (numer of channels , width =1 )
        self.K = int(K) # the number of output nodes of the net (number of actions)
        
        # The class of the loss function 
        self.net = net
        # self.loss_func = tf.keras.losses.Huber()
        self.loss_func = tf.keras.losses.MSE
        
        # Optimizer
        if scope == 'model':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)#0.0001)
            print('DQN main net created!')
        elif scope == 'Traget_model':
            print("Traget DQN net created!")
            
    @tf.function     
    def forward(self, Z):
        print(Z.shape)
        Z = self.net.forward(Z)
        return Z
    @tf.function
    def predict(self, x):
        # print("predict:", x.shape)
        # x is the satate in our case
        return self.forward(x)
    
    # @tf.function
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            # print(x)
            x = tf.expand_dims(x, axis = 0)
            # x = np.expand_dims(x, axis = 0)
            x = x.numpy()
            x = np.float32(x)
            # x = tf.stop_gradient(x)
            a_possible = self.predict(x)
            return np.argmax(a_possible[0])
        
    #@tf.function
    def cost(self, s, actions, G, idxes, weights):
        """
        Parameters
        ----------
        s : states --> for predicting the Q(s,a)
        actions : actions took by the agent 
        G : future return.
        """
        prediction = self.forward(s) # s is the states, predictions is the Q(a,s)
        # now we take into account the action values (Q(s,a)) which corresponds to the action took in the past 
        # print(actions)
        # input()
        predicted_q_s_a = prediction * tf.one_hot(actions, self.K) #elemwnt wise multiplication 
        
        selected_action_values = tf.reduce_sum(predicted_q_s_a, axis = [1])
        abs_delta = np.abs(G - selected_action_values) ## without margine 

        costi = tf.reduce_mean(
                                self.loss_func(y_true = G , y_pred = selected_action_values) * weights
                                )*0.5

        return costi, idxes, abs_delta 
    
    #@tf.function
    def update_weights(self, states, actions, targets, idxes, weights = None):
        # print("actions:", actions)
        with tf.GradientTape(watch_accessed_variables = True) as tape:
            cost_i, idxes,abs_delta  = self.cost(states, actions, targets,idxes, weights)
            
        gradients = tape.gradient(cost_i, self.net.trainable_params)
        
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_params))
        
        return cost_i, idxes,abs_delta
    
    # @tf.function
    def copy_params_from(self, model):
        for i in range(len(self.net.trainable_params)):
            self.net.trainable_params[i].assign(model.net.trainable_params[i].numpy())
            
    
    def save(self, name):
        
        if not os.path.isdir('Weights'):
            os.mkdir('Weights')
            
        params = [p.numpy() for p in self.net.trainable_params]
        file_name = 'wights/' + name +'.pickle'
        
        with open(file_name, "wb") as file:
            pickle.dump(params, file)
            
        print('Saveing weights done!')
    
    def load(self, path):
        
        if not os.path.isfile(path):
            print("there is no such file! Check your path.")
            print("Given path: %s" % path)
            
        with open(path, "rb") as file:
            params  = pickle.load(file)
            
        for tp, p in zip(self.net.trainable_params,params):
            tp.assign(p)
        
        print("Weights loaded successfully!")
    

            

        
class DDQN(object):
    
    def __init__(self, net,number_of_actions , learning_rate, gamma, n_steps = 1):#model, target_model, replay_memory  = ReplayMemory(), r=4):
        #K < - - number of actions we can Take 
        self.main_net = DQN(net = net,  K = number_of_actions ,learning_rate = learning_rate, scope = 'model' )
        self.target_model = DQN(net = net, K = number_of_actions, scope = 'Traget_model')
        self.gamma = gamma
        self.batch_sz = 32
        self.n_steps = n_steps
        
    # @tf.function
    def learn(self, experience_replay_buffer):
        # sample experiences
        # states, actions, rewards, next_states, dones, weights = experience_replay_buffer.get_minibatch()
        states, actions, rewards, next_states, dones, idxes, weights = experience_replay_buffer.get_minibatch()

        next_Qs = self.target_model.predict(next_states).numpy() # in R(n*Num_actions)
        next_Qs_main_net = self.main_net.predict(next_states).numpy()
 
        #next_Qs = self.target_model.predict(next_states) <--- use this if you want to do the next line , as in https://www.nature.com/articles/nature14236.pdf
        # # next_Q  = np.amax(next_Qs, axis = 1) <--- this approach is described in https://www.nature.com/articles/nature14236.pdf (dqn with experience replay)
        #next_Q = [next_Qs[i][actions[i]] for i in range(len(actions))] ## as decribed in https://arxiv.org/pdf/1509.06461.pdf (original DDQN paper) to avoid over estimating (read that paper !!)
        next_Q = [next_Qs[i][np.argmax(next_Qs_main_net[i])] for i in range(len(actions))] # alittle bit different
        
        # time.sleep(1000)
        targets = rewards + np.invert(dones).astype(np.float32) * self.gamma**(self.n_steps) * next_Q
        
        # Update model
        cost_i,  idxes, abs_delta = self.main_net.update_weights(states, actions, targets, idxes, weights)
        return cost_i, idxes, abs_delta

    def sample_action(self, state, eps = 0, training = None):
        # print(state.shape, "DDQN")
        action = self.main_net.sample_action(state, eps = eps)
        return action 
    
    def save_weights(self):#, name ):
        if not os.path.isdir('weights_ddqn'):
            os.makedirs('weights_ddqn')
        params = [param.numpy() for param in self.main_net.net.trainable_params]
        file_name = 'weights_ddqn/params_ddqn.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
            
        print('Saving wweights!')
            
        
    def load_weights(self):#,  name ):
        if os.path.isdir('weights_ddqn'):
            file_name = 'weights_ddqn/params_ddqn.pickle'
            with open(file_name, 'rb') as f:
                params = pickle.load(f)
                
            for tp,tm,p in zip(self.main_net.net.trainable_params,
                               self.target_model.net.trainable_params, params):
                tp.assign(p)
                tm.assign(p)
                
            print("weights load successfully!")
            
        else:
            print("weights load where not successed!!!")
            print('pass this loading and start randomlly')
            time.sleep(10)
    
    def update_target_weights(self):
        for w_target, w_main_net in zip(self.target_model.net.trainable_params, 
                                        self.main_net.net.trainable_params):
            w_target.assign(w_main_net.numpy())
            
    def load_given_weights(self, w):
        for tp,tm,p in zip(self.main_net.net.trainable_params,
                               self.target_model.net.trainable_params, w):
            
            tp.assign(p)
            tm.assign(p)
    
    def get_model_weights(self):
        weights = [w.numpy() for w in self.main_net.net.trainable_params]
        return weights