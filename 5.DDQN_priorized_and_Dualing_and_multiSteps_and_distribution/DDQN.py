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
    def forward(self, Z, log=False):
        print("Z:",Z.shape)
        q_s_a, cliped_dist = self.net(Z, log = log)
        return q_s_a, cliped_dist
    
    @tf.function
    def predict(self, x):
        # print("predict:", x.shape)
        # x is the satate in our case
        q_s_a, cliped_dist = self.forward(x)
        return q_s_a, cliped_dist
    
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
            q_s_a, cliped_dist= self.predict(x)
            
            return np.argmax(q_s_a[0])
        
    #@tf.function
    def cost(self, s, actions, G, idxes, weights):
        """
        Parameters
        ----------
        s : states --> for predicting the Q(s,a)
        actions : actions took by the agent 
        G : future return.
        """
        
        
        
        # log_ps = tf.constant([[0.1, 0.2, 0.3, 0.4],
        #               [0.5, 0.6, 0.7, 0.8],
        #               [0.9, 1.0, 1.1, 1.2]], dtype=tf.float32)
        # batch_size = 3
        # actions = tf.constant([1, 0, 3], dtype=tf.int32)
        
        # # Create indices for tf.gather_nd
        # indices = tf.stack([tf.range(batch_size), actions], axis=1)
        
        # # Use tf.gather_nd to select elements from log_ps
        # selected_elements = tf.gather_nd(log_ps, indices)
        
        batch_size = len(s)
        q_s_a, log_ps = self.forward(s,log= True) # Q =(N, K), log_p = (N, K, atoms)
        indices = tf.stack([tf.range(batch_size), actions], axis=1)
        log_ps_a = tf.gather_nd(log_ps, indices) #  log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)
        # now we take into account the action values (Q(s,a)) which corresponds to the action took in the past 

        
        cost = -tf.math.reduce_sum(G * log_ps_a, axis = 1) # (N,1) # cross-entropy
         
        abs_delta = cost.numpy() # also can be done using tf.stop_gradient(cost)
         
        costi = tf.reduce_mean(weights * cost)
        
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
    
    def __init__(self, net,number_of_actions , learning_rate, gamma, num_atoms, support,
                 args , n_steps = 1):#model, target_model, replay_memory  = ReplayMemory(), r=4):
        #K < - - number of actions we can Take 
        self.main_net = DQN(net = net,  K = number_of_actions ,learning_rate = learning_rate, scope = 'model' )
        self.target_model = DQN(net = net, K = number_of_actions, scope = 'Traget_model')
        self.gamma = gamma
        self.batch_sz = 32
        self.n_steps = n_steps
        self.atoms = num_atoms
        self.support = support # tf.linspace(args.V_min, args.V_max, self.atoms)
        self.Vmin = args.Vmin
        self.Vmax = args.Vmax
        self.delta_z = self.support[1]-self.support[0]
        # self.delta_z = (args.Vmax - args.Vmin)/(num_atoms-1)
    # @tf.function
    def learn(self, experience_replay_buffer):
        # sample experiences
        # states, actions, rewards, next_states, dones, weights = experience_replay_buffer.get_minibatch()
        states, actions, rewards, next_states, dones, idxes, weights = experience_replay_buffer.get_minibatch()

        q_s_a_target, ps_target = self.target_model.predict(next_states)
        q_s_a_main, ps_main = self.main_net.predict(next_states)
        # next_Qs = self.target_model.predict(next_states).numpy() # in R(n*Num_actions)
        # next_Qs_main_net = self.main_net.predict(next_states).numpy()
        # self.support = np.array([[1],[2],[3]]) #  a = np.linspace(0, 100, 3)
        
        # dns = self.support* ps_main # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
        # q_s_a_main = tf.reduce_sum(dns, axis = 2)
        # # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
        selected_actions = tf.argmax(q_s_a_main, axis =1 ).numpy()
        
        batch_size = len(states)
        # q_s_a_target, ps_target  # Probabilities p(s_t+n, ·; θtarget)
        # pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
        indices = tf.stack([tf.range(batch_size), selected_actions], axis=1)
        ps_target_a = tf.gather_nd(ps_target, indices) #  ps[range(self.batch_size), actions],  
        # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
        
        # Compute Tz (Bellman operator T applied to z)
        # Tz = R^n + (γ^n)z (accounting for terminal states)
        # Tz = np.expand_dims(rewards, axis = 1) + np.expand_dims(np.invert(dones).astype(np.float32), axis= 1) *\
        #     discount**(n_steps) * np.expand_dims(np.squeeze(support), axis = 0)
            
        Tz = np.expand_dims(rewards, axis = 1) + np.expand_dims(np.invert(dones).astype(np.float32), axis= 1) *\
                self.gamma**(self.n_steps) * np.expand_dims(np.squeeze(self.support), axis = 0)
        # Tz = rewards + np.invert(dones).astype(np.float32) * self.discount**(self.n_steps) * self.support
        Tz = tf.clip_by_value(Tz, clip_value_min = self.Vmin, clip_value_max = self.Vmax)
        # Tz = tf.clip_by_value(Tz, clip_value_min = 50 , clip_value_max = 150)
        # Compute L2 projection of Tz onto fixed support z
        
        # b = (Tz - Vmin) / delta_z
        b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
        b = np.array(b)
        l, u = np.floor(b), np.ceil(b)
        
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        # u[(l < (atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        m = tf.zeros(shape = (batch_size, self.atoms))
        m = tf.zeros(shape = batch_size * self.atoms)
        # m = tf.zeros(shape = (batch_size, atoms))
        # Create an offset tensor using TensorFlow operations
        offset = tf.linspace(0.0, float((batch_size - 1) * self.atoms), batch_size)  # Create a linear space of values
        
        offset = tf.expand_dims(offset, axis=1)  # Add a new dimension to 'offset' (shape: batch_size x 1)
        
        offset = tf.tile(offset, [1, self.atoms])  # Tile 'offset' to match the shape (batch_size, atoms)
        
        indexes_l =tf.reshape(l+offset, [-1])
        indexes_l = tf.cast(indexes_l, dtype = tf.int32)
        values_l = tf.reshape(ps_target_a * (u - b), [-1])
        
        m = tf.tensor_scatter_nd_add(m, tf.expand_dims(indexes_l, axis=1), values_l)
        
        indexes_u = tf.cast(tf.reshape(u + offset, [-1]), dtype=tf.int32)
        values_u = tf.reshape(ps_target_a * (b - l), [-1])
        
        m = tf.tensor_scatter_nd_add(m, tf.expand_dims(indexes_u, axis=1), values_u)
        
        m = tf.reshape(m ,shape = (batch_size, self.atoms) )
        # Assuming 'actions' is a TensorFlow tensor, convert 'offset' to the same data type and device
        # offset = tf.cast(offset, dtype=actions.dtype)  #
        targets = np.array(tf.stop_gradient(m))
        #next_Qs = self.target_model.predict(next_states) <--- use this if you want to do the next line , as in https://www.nature.com/articles/nature14236.pdf
        # # next_Q  = np.amax(next_Qs, axis = 1) <--- this approach is described in https://www.nature.com/articles/nature14236.pdf (dqn with experience replay)
        # #next_Q = [next_Qs[i][actions[i]] for i in range(len(actions))] ## as decribed in https://arxiv.org/pdf/1509.06461.pdf (original DDQN paper) to avoid over estimating (read that paper !!)
        # next_Q = [next_Qs[i][np.argmax(next_Qs_main_net[i])] for i in range(len(actions))] # alittle bit different
        
        # # time.sleep(1000)
        # targets = rewards + np.invert(dones).astype(np.float32) * self.gamma**(self.n_steps) * next_Q
        
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