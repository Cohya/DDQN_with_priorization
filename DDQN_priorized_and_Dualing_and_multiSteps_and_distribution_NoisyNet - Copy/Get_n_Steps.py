

# minor modified from https://github.com/medipixel/rl_algorithms/blob/master/algorithms/common/helper_functions.py
def get_n_step_info(n_step_buffer, gamma):
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_step_buffer[-1][-3:] # Only the last transition 

    for transition in reversed(list(n_step_buffer)[:-1]): ## loop for previous transitions
        r, n_s, d = transition[-3:]
        
        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done