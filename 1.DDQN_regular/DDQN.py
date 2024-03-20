import numpy as np
import random  
import tensorflow as tf 
import os 
import pickle 
import time 

class ReplayMemory(object):
    
    def __init__(self, capacity, number_of_channels , agent_history_length = 4, batch_size = 32):
        self.batch_size = batch_size
        self.size = capacity
        self.current = 0
        self.count = 0
        self.width = 1 # the width of the vector 
        self.agent_history_length = agent_history_length
        self.number_of_channels = number_of_channels
        # Pre-allocate memory 
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype = np.float32)
        self.observations = np.empty(shape  = (self.size, self.number_of_channels,self.width), dtype = np.int32)
        self.terminal_flags =  np.empty(self.size, dtype = np.bool)
        
        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty(shape = (self.batch_size, self.agent_history_length, self.number_of_channels, self.width), dtype = np.int32)
        self.new_states = np.empty(shape = (self.batch_size, self.agent_history_length, self.number_of_channels, self.width), dtype= np.int32)
        self.indices = np.empty(self.batch_size, dtype = np.int32)
    
    
    def add_experience(self, action, observation, reward, terminal):
        
        if observation.shape != (self.number_of_channels,1):
            print("Observation shape:", observation.shape)
            raise ValueError('Dimension of observation is wrong!')
            
        
        self.actions[self.current] = action 
        self.observations[self.current,...] = observation
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+ 1)
        self.current = (self.current + 1) % self.size
        
        
    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
            
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min %s" % str(self.agent_history_length - 1))
        s = self.observations[index - self.agent_history_length + 1: index + 1, ...]
        return s
    
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True :
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue 
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue #  # it is a circular so this tuple is not a valid state
                    
                if self.terminal_flags[index - self.agent_history_length: index].any():
                    continue # we check here that we are not at the boundary of an episode
                    
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
            
        self._get_valid_indices()
        
        for i, idx in enumerate(self.indices):
         
            self.states[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
            
        return np.transpose(self.states, axes = (0,2,3,1)), self.actions[self.indices], \
                self.rewards[self.indices], \
                    np.transpose(self.new_states, axes = (0,2,3,1)),\
                    self.terminal_flags[self.indices]
            
            
            
            
        
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
        
    @tf.function
    def cost(self, s, actions, G):
        """
        Parameters
        ----------
        s : states --> for predicting the Q(s,a)
        actions : actions took by the agent 
        G : future return.
        """
        prediction = self.forward(s) # s is the states, predictions is the Q(a,s)
        # now we take into account the action values (Q(s,a)) which corresponds to the action took in the past 
        predicted_q_s_a = prediction * tf.one_hot(actions, self.K) #elemwnt wise multiplication 
        
        selected_action_values = tf.reduce_sum(predicted_q_s_a, axis = [1])
     

        costi = tf.reduce_mean(
                                    self.loss_func(y_true = G , y_pred = selected_action_values) 
                                    )*0.5
        # print(costi)
        return costi 
    
    @tf.function
    def update_weights(self, states, actions, targets):
        
        with tf.GradientTape(watch_accessed_variables = True) as tape:
            cost_i  = self.cost(states, actions, targets)
            
        gradients = tape.gradient(cost_i, self.net.trainable_params)
        
        self.optimizer.apply_gradients(zip(gradients, self.net.trainable_params))
        
        return cost_i
    
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
    
    def __init__(self, net,number_of_actions , learning_rate, gamma):#model, target_model, replay_memory  = ReplayMemory(), r=4):
        #K < - - number of actions we can Take 
        self.main_net = DQN(net = net,  K = number_of_actions ,learning_rate = learning_rate, scope = 'model' )
        self.target_model = DQN(net = net, K = number_of_actions, scope = 'Traget_model')
        self.gamma = gamma
        self.batch_sz = 32
        
        
    # @tf.function
    def learn(self, experience_replay_buffer):
        # sample experiences
        # states, actions, rewards, next_states, dones, weights = experience_replay_buffer.get_minibatch()
        states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()
        # print(states.dtype)
        # print(next_states.dtype)
        # Calculate targets expected future rewards
        
        next_Qs = self.target_model.predict(next_states).numpy() # in R(n*Num_actions)
        next_Qs_main_net = self.main_net.predict(next_states).numpy()
        # print(next_Qs)
        #print("actions:", actions)
        # print("asdasd")
        #print(next_Qs_main_net[0])
        #next_Qs = self.target_model.predict(next_states) <--- use this if you want to do the next line , as in https://www.nature.com/articles/nature14236.pdf
        # # next_Q  = np.amax(next_Qs, axis = 1) <--- this approach is described in https://www.nature.com/articles/nature14236.pdf (dqn with experience replay)
        #next_Q = [next_Qs[i][actions[i]] for i in range(len(actions))] ## as decribed in https://arxiv.org/pdf/1509.06461.pdf (original DDQN paper) to avoid over estimating (read that paper !!)
        next_Q = [next_Qs[i][np.argmax(next_Qs_main_net[i])] for i in range(len(actions))] # alittle bit different
        # print(next_Q)
        # time.sleep(1000)
        targets = rewards + np.invert(dones).astype(np.float32) * self.gamma * next_Q
        
        # Update model
        cost_i = self.main_net.update_weights(states, actions, targets)
        
        return cost_i

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