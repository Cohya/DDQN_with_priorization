
import gym 
from DDQN import DDQN

from Nets_keras import AnnDualing_Opt1, AnnDualing_Opt2
from Agent  import Agent
from ProportionalPriorizationReplayBuffer import ProportionalPriorizationReplayBuffer
import numpy as np 
import tensorflow as tf 
import pickle 
import matplotlib.pyplot as plt 
from collections import deque # Doubly Ended Queue data structure 
from Get_n_Steps import get_n_step_info

    
np.random.seed(1)
tf.random.set_seed(1)


env =  gym.make('CartPole-v1')

D = len(env.observation_space.sample())
K = env.action_space.n
K = env.action_space.n
net = AnnDualing_Opt2(input_shape = D, K = K )

n_steps = 3
n_step_buffer = deque(maxlen = n_steps)

ddqn = DDQN(net = net,
            number_of_actions= K,
            learning_rate = 0.001,
            gamma = 0.99, 
            n_steps = n_steps)

gamma = 0.99
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
N  = capacity * 4#  Number of total game to train on (Episodes)
max_iteration_of_episode = 1000 # was 1000




for n in range(N):
    if n % 200 == 0:
        print(n, "/", N)
    observation = env.reset()[0] # (4,)
    done = False
    totalrewards = 0
    # iters = 0
    iter_internal_game = 0
    # print(n)
    while not done and iter_internal_game < 2000:
        # if we reach 2000, just quit, don't want this going forever 
        # the 200 limit seems a bit early 
        
        eps = 1.0 / (1.0 + n)**(0.3) #(0.2)
        state = obs # (1,  D)
        action = agent.sample_action(state, eps = eps, training= True)
        prev_observation = observation 
        
        
        next_obs, reward, done, truncated,  info = env.step(action)
         
        totalrewards += reward
        
        
        
        if iter_internal_game >= max_iteration_of_episode:
            done = True
            # print("iter_internal_game:", iter_internal_game, n)
        
        if done and iter_internal_game < max_iteration_of_episode:
            reward = -200 ##  <--maybe it is to big now 
            # print(n, "reduce -200")
        # n-step replay buffer
        
        temp_transition = [obs, action, reward, next_obs, done]
        n_step_buffer.append(temp_transition)
        
        if len(n_step_buffer) == n_steps:  # fill the n-step buffer for the first translation
                # add a multi step transition
                # reward_G = r_1 + gamma * r_2 + ... + gamma^(n-1)r_n - gamma^n Q(s(n+1),a)
                ## we add gamma^n* Q(s(n+1), a) when we compute in the DDQN the target Q
                reward_G, next_obs_to_store_G, done_to_stor = get_n_step_info(n_step_buffer, gamma)
                obs_minus_n_steps, action_minus_n_steps = n_step_buffer[0][:2]
        else:
            next_obs_to_store_G = next_obs
            reward_G = reward
            obs_minus_n_steps = obs
            action_minus_n_steps = action
            done_to_stor = done
            
        if global_iters == 0:
            p = agent.experience_replay_buffer.p1
        else:
            p = agent.experience_replay_buffer.get_max_p() #np.max(agent.replay_buffer.tree[-self.replay_buffer.capacity:]) 

            
        ## Store it inside the Priorized RMB
        agent.experience_replay_buffer.store_transition(p, obs_minus_n_steps,
                                                        action_minus_n_steps, 
                                                        reward_G,
                                                        next_obs_to_store_G,
                                                        done_to_stor)
        

        if global_iters > batch_size * 10:
            costi, idxes, abs_delta = agent.learn()
            # Update the priorization 
            agent.experience_replay_buffer.update_priorization(idxes, abs_delta)
            cost_vec.append(costi)
            
        # iters += 1
        iter_internal_game += 1
        global_iters += 1
        
        obs = next_obs
        
        if global_iters % copy_period == 0:
            agent.model.update_target_weights()
            # here we update the "Target network" to keep stability !! 
            # we do it by copy the weights from the main NN.
            
    total_rewards_vec.append(totalrewards)

    if n % 20 == 0:
        print(f"R average 20: {np.mean(total_rewards_vec[-20:])}, eps: {eps}, Episode: {n}")
        if np.mean(total_rewards_vec[-20:]) >= 1000: #and eps < 0.1 :
            break
        
plt.plot(total_rewards_vec, label = 'Accumulated rewards')
plt.ylabel("Rewards")
plt.xlabel("Episode") 
smoothed_r =[]
for i in range(len(total_rewards_vec)- 50):
    smoothed_r.append(np.mean(total_rewards_vec[i:i+50]))
    
plt.plot(smoothed_r, label = 'smoothed accumulated rewards 100')
plt.legend()

# with open("total_rewards_vec.pk", "wb") as file:
#       pickle.dump(total_rewards_vec, file)  
     
# trained_weights = agent.get_model_weights()

# with open("train_weights_DDQN_dualing.pk", "wb") as file:
#     pickle.dump(trained_weights, file)

# with open("train_weights_DDQN_dualing.pk", "rb") as file:
#     trained_weights_loded = pickle.load(file)
    
# agent.load_given_weights(trained_weights_loded)

r_test = []
for _ in range(100):
    obs = env.reset()[0]
    done = False
    r_episode  = 0
    iter_internal_game = 0 
    while not done:
        state =obs #(1,4,1) ( history, D, channels)
        a = agent.sample_action(x = state, eps=0, training = False)
        
        next_observation, reward, done, truncated,  info= env.step(action = a)
        iter_internal_game += 1
        if iter_internal_game > max_iteration_of_episode:
            done = True
        
        r_episode += reward
        
        obs = next_observation
    r_test.append(r_episode)

print("r_test average value:", np.mean(r_test))