
import gym 
from DDQN import DDQN

from Nets_keras import Ann
from Agent  import Agent
from ProportionalPriorizationReplayBuffer import ProportionalPriorizationReplayBuffer
import numpy as np 
import tensorflow as tf 
import pickle 
import matplotlib.pyplot as plt 

np.random.seed(1)
tf.random.set_seed(1)


env =  gym.make('CartPole-v1')

D = len(env.observation_space.sample())
K = env.action_space.n
K = env.action_space.n
net = Ann(input_shape = D, K = K )

ddqn = DDQN(net = net,
            number_of_actions= K,
            learning_rate = 0.001,
            gamma = 0.99)

batch_size = 32
capacity = 1000
replay_memory = ProportionalPriorizationReplayBuffer(capacity =capacity, obs_dims= D)
agent = Agent(model = ddqn, 
              experience_replay_buffer = replay_memory)

## Before training 
obs = env.reset()[0]
done = False
r_episode  = 0

while not done:
    state =obs 
    a = agent.sample_action(x = state, eps=0, training = False)
    
    next_obs, r, done, _, _ = env.step(action = a)
    
    r_episode += r 
    
print("Results before training:", r_episode)
  
global_iters = 0
copy_period = 50 
total_rewards_vec = []
cost_vec = []
N  = capacity * 4
max_iteration_of_episode = 1000
for n in range(N):
    if n % 200 == 0:
        print(n, "/", N)
    observation = env.reset()[0] # (4,)
    done = False
    totalrewards = 0
    iters = 0
    iter_internal_game = 0
    
    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever 
        # the 200 limit seems a bit early 
        
        eps = 1.0 / (1.0 + n)**(0.3) #(0.2)
        state = obs # (1,  D)
        action = agent.sample_action(state, eps = eps, training= True)
        prev_observation = observation 
        
        next_obs, reward, done, truncated,  info = env.step(action)
        
        if global_iters == 0:
            p = agent.experience_replay_buffer.p1
        else:
            p = agent.experience_replay_buffer.get_max_p() #np.max(agent.replay_buffer.tree[-self.replay_buffer.capacity:]) 
            # print("sdfsdf")
        
        totalrewards += reward
        iter_internal_game += 1
        if iter_internal_game > max_iteration_of_episode:
            done = True
        
        if done and iters< max_iteration_of_episode:
            reward = -200
            
        ## Store it inside the Priorized RMB
        agent.experience_replay_buffer.store_transition(p, obs, action, reward, next_obs, done)
        # print('global_iters:', global_iters)
        # input("g:")
        if global_iters > batch_size * 10:
            costi, idxes, abs_delta = agent.learn()
            # Update the priorization 
            agent.experience_replay_buffer.update_priorization(idxes, abs_delta)
            cost_vec.append(costi)
            
        iters += 1
        global_iters += 1
        
        obs = next_obs
        
        if global_iters % copy_period == 0:
            agent.model.update_target_weights()
            # here we update the "Target network" to keep stability !! 
            # we do it by copy the weights from the main NN.
            
    total_rewards_vec.append(totalrewards)

    if n % 20 == 0:
        print(f"R average 20: {np.mean(total_rewards_vec[-20:])}, eps: {eps}, Episode: {n}")
        # if np.mean(total_rewards_vec[-20:]) >= 199:
        #     break
        
plt.plot(total_rewards_vec, label = 'Accumulated rewards')
plt.ylabel("Rewards")
plt.xlabel("Episode") 
smoothed_r =[]
for i in range(len(total_rewards_vec)- 100):
    smoothed_r.append(np.mean(total_rewards_vec[i:i+100]))
    
plt.plot(smoothed_r, label = 'smoothed accumulated rewards 100')
plt.legend()

# with open("total_rewards_vec.pk", "wb") as file:
#       pickle.dump(total_rewards_vec, file)  
     
# trained_weights = agent.get_model_weights()

# with open("train_weights_DDQN.pk", "wb") as file:
#     pickle.dump(trained_weights, file)

# with open("train_weights_DDQN.pk", "rb") as file:
#     trained_weights_loded = pickle.load(file)
    
# agent.load_given_weights(trained_weights_loded)

# r_test = []
# for _ in range(100):
#     obs = env.reset()[0]
#     done = False
#     r_episode  = 0
#     iter_internal_game = 0 
#     while not done:
#         state =obs #(1,4,1) ( history, D, channels)
#         a = agent.sample_action(x = state, eps=0, training = False)
        
#         next_observation, reward, done, truncated,  info= env.step(action = a)
#         iter_internal_game += 1
#         if iter_internal_game > max_iteration_of_episode:
#             done = True
        
#         r_episode += r
        
#         obs = next_observation
#     r_test.append(r_episode)

# print("r_test average value:", np.mean(r_test))