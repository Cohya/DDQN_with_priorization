import tensorflow as tf 
class Agent(object):
    
    def __init__(self, model, experience_replay_buffer, i_d = None,
                 verbose = False):
        
        # model == Agent brain
        self.model = model
        self.experience_replay_buffer = experience_replay_buffer
        self.current_channel = None 
        self.old_channel = None
        self.verbose = verbose
        self.change_channel_counter = -1
        self.i_d = i_d
        self.net_location = None
        
    def update_current_channel(self, channel):
        if self.current_channel is not None:
            if self.old_channel != channel: 
                self.old_channel = int(self.current_channel)
                self.change_channel_counter += 1
        self.current_channel = channel
        
    def sample_action(self, x, eps,training):
        action = self.model.sample_action(x, eps, training)
        self.action = action
        return action
    
    def learn(self): # experience_replay_buffer
        cost = self.model.learn(self.experience_replay_buffer)#,
                              #  self.experience_replay_buffer.batch_size)

        return cost
    
    def save_weights(self):
        self.model.save_weights()

        
    def __load_weights(self):
        self.model.load_weights()
        
    
    def load_given_weights(self, w):
        self.model.load_given_weights(w)
        if self.verbose: 
            print("Given weights were loaded!")
      
    def get_model_weights(self):
        weights = self.model.get_model_weights()
        return weights 
    
  
    