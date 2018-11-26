import gym
from algos import *
from rollouts import *
import tensorflow as tf
import numpy as np
import pickle
from envs import *
register_custom_envs()
from envs.pointmass import PointMass
from envs.r7dof import R7DOFEnv
from envs.r7dof_xyz import R7DOFEnv as R7DOFxyzEnv
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def train_mil(
    n_iters, save_dir, name,
    envs, expert_trajs,
    timesteps_per_rollout=200, ep_max_len=20,
    metarl_algo=MetaRL, use_checkpoint=False, checkpoint_iter=0,
):
    tf.reset_default_graph()
    env_fns = {k: lambda: v for (k, v) in envs.items()}
    if use_checkpoint:
        checkpoint = '{}/{}_model_{}'.format(save_dir, name, checkpoint_iter)
    else:
        checkpoint = None
    expert_model = metarl_algo(name, env_fns, expert_trajs, checkpoint=checkpoint, save_path='{}/{}_model'.format(save_dir, name))

    print('\nTraining expert...')
    expert_model.train(n_iters, timesteps_per_rollout, ep_max_len)

    return expert_model

def test_mil(expert_dir, expert_name, checkpoint_iter,
    envs, expert_trajs,
    timesteps_per_rollout=200, ep_max_len=20,
    n_test_rollouts=5,
    metarl_algo=MetaRL
):
    for (task, env) in envs.items():
        print('Task:', task)
        tf.reset_default_graph()
        env_fns = {task: lambda: env}
        expert_traj = {task: expert_trajs[task]}
        expert_model = metarl_algo(expert_name, env_fns, expert_trajs=expert_traj, meta_batch_size=1, checkpoint='{}/{}_model_{}'.format(expert_dir, expert_name, checkpoint_iter))
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
            try:
                env.close()
            except Exception as e:
                pass
            time.sleep(1)
        print('avg ep reward:', tot_reward / n_test_rollouts)

if __name__ == '__main__':
    ### VISION MIL
    #expert_trajs = pickle.load(open('data/r7dof/expert_trajs.pkl', 'rb'))
    #envs = {k: R7DOFEnv(k) for (k, v) in expert_trajs.items()}
    #train_mil(
    #    n_iters=5000, save_dir='data/r7dof', name='vision_MIL',
    #    envs=envs, expert_trajs=expert_trajs,
    #    timesteps_per_rollout=600, ep_max_len=30,
    #    use_checkpoint=True, checkpoint_iter=100,
    #)
    #test_mil(
    #    expert_dir='data/r7dof', expert_name='vision_MIL', checkpoint_iter=4610,
    #    envs=envs, timesteps_per_rollout=600, ep_max_len=30
    #)

    ### NOVISION R7DOF
    #TODO: recollect trajs
    #expert_trajs = pickle.load(open('data/r7dof/expert_trajs.pkl', 'rb'))
    #for task in expert_trajs.keys():
    #    expert_trajs[task]['obs'] = expert_trajs[task]['obs'][:, 64*64*3:]
    #envs = {k: R7DOFEnv(k) for (k, v) in expert_trajs.items()}
    #train_mil(
    #    n_iters=500, save_dir='data/r7dof', name='r7dof_novision_MIL',
    #    envs=envs, expert_trajs=expert_trajs,
    #    timesteps_per_rollout=600, ep_max_len=30,
    #    use_checkpoint=False,
    #)
    #test_mil(
    #    expert_dir='data/r7dof', expert_name='r7dof_novision_MIL', checkpoint_iter=499,
    #    envs=envs, timesteps_per_rollout=600, ep_max_len=30
    #)

    ### VISION BC
    #expert_trajs = pickle.load(open('data/r7dof/expert_trajs_test.pkl', 'rb'))
    #envs = {k: R7DOFEnv(k) for (k, v) in expert_trajs.items()}
    #train_mil(n_iters=5000, save_dir='data/r7dof', name='vision_BC_test', envs=envs, expert_trajs=expert_trajs, timesteps_per_rollout=1, ep_max_len=1)
    #test_mil(expert_dir='data/r7dof', expert_name='vision_BC_test', envs=envs, timesteps_per_rollout=30, ep_max_len=30)

    ### VISUAL POINTMASS
    #expert_trajs = pickle.load(open('data/r7dof/expert_trajs_xyz.pkl', 'rb'))
    #envs = {k: R7DOFxyzEnv(k) for (k, v) in expert_trajs.items()}
    #train_mil(
    #    n_iters=500, save_dir='data/r7dof', name='xyz_vision_MIL',
    #    envs=envs, expert_trajs=expert_trajs,
    #    timesteps_per_rollout=600, ep_max_len=30,
    #    use_checkpoint=False,
    #)
    #test_mil(
    #    expert_dir='data/r7dof', expert_name='xyz_vision_MIL', checkpoint_iter=499,
    #    envs=envs, timesteps_per_rollout=600, ep_max_len=30
    #)

    ### POINTMASS
    #train_expert_trajs = pickle.load(open('data/pointmass/train_expert_trajs.pkl', 'rb'))
    #train_envs = {k: PointMass(np.array(k)) for (k, v) in train_expert_trajs.items()}
    #train_mil(n_iters=100, save_dir='data/pointmass', name='MIL', envs=train_envs, expert_trajs=train_expert_trajs)

    test_expert_trajs = pickle.load(open('data/pointmass/test_expert_trajs.pkl', 'rb'))
    test_envs = {k: PointMass(np.array(k)) for (k, v) in test_expert_trajs.items()}
    test_mil(expert_dir='data/pointmass', expert_name='MIL', checkpoint_iter=99, envs=test_envs, expert_trajs=test_expert_trajs)
