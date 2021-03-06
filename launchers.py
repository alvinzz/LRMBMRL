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
    timesteps_per_rollout=200, ep_max_len=20,
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
    expert_model.train(n_iters, timesteps_per_rollout, ep_max_len)

    expert_model.saver.save(expert_model.sess, '{}/{}_model'.format(save_dir, name))
    return expert_model

def visualize_expert(env_name, expert_dir, expert_name, rl_algo=RL, ep_max_len=20, n_runs=1):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    expert_model = rl_algo(expert_name, env_fn, checkpoint='{}/{}_model'.format(expert_dir, expert_name))
    env = gym.make(env_name)
    tot_reward = 0
    for n in range(n_runs):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < ep_max_len:
            last_obs = obs
            env.render()
            time.sleep(0.02)
            action = expert_model.policy.act([obs], expert_model.sess)[0]
            obs, reward, done, info = env.step(action)
            tot_reward += reward
            t += 1
        time.sleep(1)
    print('avg ep reward:', tot_reward / n_runs)

def train_MBR(
    save_dir, name, env_name, dataset,
    repr_algo=ModelLearning, use_checkpoint=False,
):
    tf.reset_default_graph()
    env_fn = lambda: gym.make(env_name)
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    MBR_model = repr_algo(name, env_fn, dataset, checkpoint=checkpoint)

    print('\nTraining Model-Based Representation...')
    MBR_model.train()

    MBR_model.saver.save(MBR_model.sess, '{}/{}_model'.format(save_dir, name))
    return MBR_model

if __name__ == '__main__':
    # train_expert(n_iters=100, save_dir='data/pointmass', name='expert', env_name='PointMass-v0')
    # visualize_expert('PointMass-v0', 'data/pointmass', 'expert', n_runs=5)

    # from utils import collect_random_dataset
    # collect_random_dataset('PointMass-v0')

    dataset = pickle.load(open('PointMass-v0_random_dataset.pkl', 'rb'))
    train_MBR('data/pointmass', 'mbr_no_reconstr', 'PointMass-v0', dataset)
