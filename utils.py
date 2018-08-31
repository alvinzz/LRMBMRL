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

def collect_random_dataset(env_name, n_trials=1000, save_file=None):
    if not save_file:
        save_file = '{}_random_dataset.pkl'.format(env_name)
    env = gym.make(env_name)
    obs, actions, rewards, next_obs = [], [], [], []
    for trial in range(n_trials):
        done = False
        ob = env.reset()
        while not done:
            obs.append(ob)
            action = env.action_space.sample()
            actions.append(action)
            ob, reward, done, _ = env.step(action)
            rewards.append([reward])
            next_obs.append(ob)
        obs.pop()

    data_dict = {
        'obs': obs,
        'actions': actions,
        'next_obs': next_obs,
        'rewards': rewards
    }
    pickle.dump(data_dict, open(save_file, 'wb'))
    return data_dict

if __name__ == '__main__':
    collect_random_dataset('PointMass-v0')
