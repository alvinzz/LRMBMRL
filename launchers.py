import gym
from algos import *
from rollouts import *
import tensorflow as tf
import numpy as np
import pickle
from envs import *
register_custom_envs()
from envs.pointmass import PointMass
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def train_mil(
    n_iters, save_dir, name,
    envs, expert_trajs,
    timesteps_per_rollout=200, ep_max_len=20,
    metarl_algo=MetaRL, use_checkpoint=False
):
    tf.reset_default_graph()
    env_fns = {k: lambda: v for (k, v) in envs.items()}
    if use_checkpoint:
        checkpoint = '{}/{}_model'.format(save_dir, name)
    else:
        checkpoint = None
    expert_model = metarl_algo(name, env_fns, expert_trajs, checkpoint=checkpoint)

    print('\nTraining expert...')
    expert_model.train(n_iters, timesteps_per_rollout, ep_max_len)

    expert_model.saver.save(expert_model.sess, '{}/{}_model'.format(save_dir, name))
    return expert_model

def test_mil(expert_dir, expert_name,
    envs, timesteps_per_rollout=200, ep_max_len=20,
    n_test_rollouts=5,
    metarl_algo=MetaRL):
    for (task, env) in envs.items():
        tf.reset_default_graph()
        env_fns = {task: lambda: env}
        expert_model = metarl_algo(expert_name, env_fns, expert_trajs={}, checkpoint='{}/{}_model'.format(expert_dir, expert_name))
        expert_model.test_update(timesteps_per_rollout, ep_max_len)
        task_env = env
        tot_reward = 0
        for n in range(n_test_rollouts):
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
        print('avg ep reward:', tot_reward / n_test_rollouts)

if __name__ == '__main__':
    train_expert_trajs = pickle.load(open('data/pointmass/train_expert_trajs.pkl', 'rb'))
    train_envs = {k: PointMass(np.array(k)) for (k, v) in train_expert_trajs.items()}
    train_mil(n_iters=100, save_dir='data/pointmass', name='MIL', envs=train_envs, expert_trajs=train_expert_trajs)

    test_expert_trajs = pickle.load(open('data/pointmass/test_expert_trajs.pkl', 'rb'))
    test_envs = {k: PointMass(np.array(k)) for (k, v) in test_expert_trajs.items()}
    test_mil(expert_dir='data/pointmass', expert_name='MIL', envs=test_envs)
