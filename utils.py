import numpy as np
import pickle
import gym
from envs import *
register_custom_envs()
from envs.pointmass import PointMass

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

def collect_expert_trajectories(env, tasks, save_path):
    data = {}
    for task in tasks:
        obs, next_obs, actions, rewards = [], [], [], []
        env = PointMass(task)
        ob = env.reset()
        for t in range(20):
            action = env.target - ob[:2]
            obs.append(ob); actions.append(action)
            ob, reward, done, info = env.step(action)
            next_obs.append(ob); rewards.append(reward)
        data[tuple(task.tolist())] = {
            'obs': np.array(obs),
            'next_obs': np.array(next_obs),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
        }
    pickle.dump(data, open(save_path, 'wb'))

if __name__ == '__main__':
    train_tasks = np.random.uniform(-1, 1, size=(40, 2))
    collect_expert_trajectories(PointMass, train_tasks, 'data/pointmass/train_expert_trajs.pkl')

    test_tasks = np.random.uniform(-1, 1, size=(10, 2))
    collect_expert_trajectories(PointMass, test_tasks, 'data/pointmass/test_expert_trajs.pkl')
