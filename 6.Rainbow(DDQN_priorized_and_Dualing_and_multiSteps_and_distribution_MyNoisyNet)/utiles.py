
import numpy as np 

def play_one(env, model, tmodel, eps, gamma, copy_period):
    global global_iters
    
    observation = env.reset()
    done = False
    totalrewards = 0
    iters = 0
    
    while not done and iters < 2000:
        # if we reach 2000, just quit, don't want this going forever 
        # the 200 limit seems a bit early 
        action = model.sample_action(observation, eps)
        prev_observation = observation 
        
        observation, reward, done, info = env.step(action)
        
        totalrewards += reward
        
        if done and iters< 200:
            reward = -200
            
        
        # update the model 
        model.add_experience(action, observation, reward, observation, done)
        costi = model.train(tmodel)
        iters += 1
        global_iters += 1
        
        if global_iters % copy_period == 0:
            tmodel.copy_params(model) # here we update the "Target network" to keep stability !! 
            # we do it by copy the weights from the main NN.
            
    return totalrewards, costi


def smoothing(x, level = 20):
    n = len(x)
    smoothed_x = []
    for i in range(n):
        if i < n - level +1: 
            smoothed_x.append(np.mean(x[i:i+level]))
        else:
            # print(i)
            smoothed_x.append(np.mean(x[i-level + 1:i + 1]))
            
    return np.array(smoothed_x)
                            
    