import numpy as np
import pickle
import gym
from envs import *
register_custom_envs()

def sample_minibatch(obs, next_obs, action_log_probs, batch_size):
    random_indices = np.random.randint(0, obs.shape[0], size=batch_size)
    return obs[random_indices], next_obs[random_indices], action_log_probs[random_indices]

def batchify(data, batch_size):
    N = data[0].shape[0]
    # batch_size = int(np.ceil(N / n_batches))
    res = []
    random_inds = np.arange(N)
    np.random.shuffle(random_inds)
    start_ind = 0
    while start_ind < N:
        batch_inds = random_inds[start_ind : min(start_ind + batch_size, N)]
        res.append([category[batch_inds] for category in data])
        start_ind += batch_size
    return res

if __name__ == '__main__':
    collect_random_dataset('PointMass-v0')
