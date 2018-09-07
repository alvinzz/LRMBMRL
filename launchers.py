import gym
from algos import *
from rollouts import *
import tensorflow as tf
import numpy as np
import pickle
from envs import *
register_custom_envs()
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def train_expert(
    n_iters, save_dir, name, env_name,
    timesteps_per_rollout=15*50, ep_max_len=50,
    rl_algo=RL, use_checkpoint=False
):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    expert_model = rl_algo(name, env_fn, checkpoint=checkpoint)

    print('\nTraining expert...')
    expert_model.train(save_dir, n_iters, timesteps_per_rollout, ep_max_len)

    return expert_model

def visualize_expert(env_name, expert_dir, expert_name, rl_algo=RL, ep_max_len=50, n_runs=1):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    expert_model = rl_algo(expert_name, env_fn, checkpoint='{}/{}_model'.format(expert_dir, expert_name))
    print('Task latent means:')
    latents = expert_model.policy.get_task_latents(expert_model.sess)
    print(latents[:, :latents.shape[1]//2])
    print('Tasks:')
    tasks = np.array(pickle.load(open('envs/pointMassRadGoals/pointMassR01Goals.pkl', 'rb'))[:15])
    print(tasks)
    env = gym.make(env_name)
    tot_reward = 0
    for n in range(n_runs):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < ep_max_len:
            last_obs = obs
            env.render()
            time.sleep(0.01)
            action = expert_model.policy.act([obs], [[env.task_id]], expert_model.sess)[0]
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            t += 1
        time.sleep(1)
    print('avg ep reward:', tot_reward / n_runs)

if __name__ == '__main__':
    train_expert(n_iters=100, save_dir='data/pointmass', name='R01', env_name='PointMassR01-v0')
    visualize_expert(env_name='PointMassR01-v0', expert_dir='data/pointmass', expert_name='R01', n_runs=5)
