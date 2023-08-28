import numpy as np 


        
# replay buffer
class SumTree:
    # little modified from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    def __init__(self, capacity):
        self.capacity = capacity    # N, the size of replay buffer, so as to the number of sum tree's leaves
        self.tree = np.zeros(2 * capacity - 1)  # equation, to calculate the number of nodes in a sum tree
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0
        
    @property
    def total_p(self):
        return self.tree[0]

    def add(self, priority, transition):
        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity
        # print("next_idx:", self.next_idx )
        
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        # print("idx:", idx)
        self.tree[idx] = priority
        self._propagate(idx, change)    # O(logn)
        # print("tree:", self.tree)
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change) ## This is updating up to the parent 

    def get_leaf(self, s):
        idx = self._retrieve(0, s)   # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    def _retrieve(self, idx, s):
        left = 2 * idx + 1 # 1
        right = left + 1 # 2
        # print("left:", left, "right:", right)
        
        if left >= len(self.tree):
            # print("out")
            return idx
        # print(self.tree[left])
        if s <= self.tree[left]:
            # print("c1")
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
        
        
class ProportionalPriorizationReplayBuffer(SumTree):
    def __init__(self,capacity, obs_dims,  alpha=0.4, beta=0.4, beta_increment_per_sample = 0.001,
                 batch_size = 32):
        super().__init__(capacity)
        self.p1 = 1 
        self.num_in_buffer = 0                      # total number of transitions stored in buffer
        self.beta = beta                            # # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
        self.beta_increment_per_sample = beta_increment_per_sample
        
        self.batch_size  = batch_size
        self.b_obs = np.empty((self.batch_size,obs_dims))
        self.b_actions = np.empty(self.batch_size, dtype=np.int8)
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty((self.batch_size, obs_dims)) 
        self.b_dones = np.empty(self.batch_size, dtype= bool)
        self.margin = 0.01                          # pi = |td_error| + margin
        
        self.abs_error_upper = 1
        self.alpha = alpha                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8] 
        
    def store_transition(self, priority, obs, action, reward, next_state, done):
        transition = [obs, action, reward, next_state, done]
        self.add(priority, transition)
        self.num_in_buffer += 1 
        
    def get_max_p(self):
        p_max = np.max(self.tree[-self.capacity:])
        return p_max
    
    # proportional prioritization sampling
    def sum_tree_sample(self):
        idxes = []
        is_weights = np.empty((self.batch_size, 1))
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        min_prob = np.min(self.tree[-self.capacity:]) / self.total_p
        max_weight = np.power(self.capacity * min_prob, -self.beta) # This is for stability 
        segment = self.total_p / self.batch_size ## To create balance to the mini-batch 
        
        for i in range(self.batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1)) ## <--this will give us different level of priority TD 
            idx, p, t = self.get_leaf(s)
            idxes.append(idx)# Store to update 
            self.b_obs[i], self.b_actions[i], self.b_rewards[i], self.b_next_states[i], self.b_dones[i] = t ##<--replace to batch 
            # P(j)
            sampling_probabilities = p / self.total_p     # where p = p ** self.alpha
            is_weights[i, 0] = np.power(self.capacity * sampling_probabilities, -self.beta) / max_weight # Fixing the bias
        return idxes, is_weights
    
    def get_minibatch(self):
        idxes, is_weights = self.sum_tree_sample()
        
        return self.b_obs, self.b_actions, self.b_rewards, self.b_next_states, self.b_dones, idxes, is_weights
    
    def update_priorization(self, idxes, abs_delta):
        clipped_error = np.where(abs_delta < self.abs_error_upper, abs_delta, self.abs_error_upper) # do not agree to take high TD error 
        ps = np.power(clipped_error, self.alpha)
        
        for idx, p in zip(idxes, ps):## The priority is already in p**alpha 
             self.update(idx, p)
             
         
                
        
        
        