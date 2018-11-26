from policies import GaussianMLPPolicy, ConvGaussianMLPPolicy
import tensorflow as tf
import numpy as np
from rollouts import *
<<<<<<< HEAD
from optimizers import *
from samplers import *
=======
from samplers import MetaParallelEnvExecutor
from optimizers import MILOptimizer
import time

from distributions import DiagGaussian

class MetaRL:
    def __init__(self,
        name,
        env_fn_dict,
        expert_trajs,
        checkpoint=None,
        save_path='model',
    ):
        with tf.variable_scope(name):
            self.env_fn_dict = env_fn_dict
            self.tasks = list(sorted(self.env_fn_dict.keys()))
            self.env_fns = [self.env_fn_dict[task] for task in self.tasks]
            self.n_tasks = len(self.tasks)
            self.expert_trajs = expert_trajs

            sample_env = self.env_fns[0]()
            self.ob_dim = sample_env.observation_space.shape[0]
            self.action_dim = sample_env.action_space.shape[0]
            # self.policy = GaussianMLPPolicy('policy', self.expert_trajs, self.ob_dim, self.action_dim, hidden_dims=[64], learn_vars=True)
            # self.policy = GaussianMLPPolicy('policy', self.expert_trajs, self.ob_dim, self.action_dim, hidden_dims=[100, 100, 100], learn_vars=True)
            self.policy = ConvGaussianMLPPolicy('policy', self.expert_trajs, self.ob_dim, self.action_dim, learn_vars=True)

            self.optimizer = MILOptimizer(self.policy, self.expert_trajs)

            self.saver = tf.train.Saver(max_to_keep=None)
            self.save_path = save_path

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, batch_timesteps=10000, max_ep_len=500, inv_save_freq=10, inv_rollout_freq=1):
        envs_per_task = int(np.ceil(batch_timesteps / max_ep_len))
        sampler = MetaParallelEnvExecutor(self.env_fns, envs_per_task, max_ep_len)
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            if iter_ % inv_rollout_freq == 0:
                # collect new, on-policy rollout
                print('Collecting rollouts...')
                start = time.time()
                mb_task_inds = np.random.choice(np.arange(len(self.expert_trajs)), size=self.meta_batch_size, replace=False)
                obs, next_obs, actions, action_log_probs, baselines, returns, rewards \
                     = collect_and_process_rollouts(sampler, self.policy, self.sess, batch_timesteps, mb_task_inds)
                end = time.time()
                print('    Done! ({}s)'.format(end-start))
            else:
                # update action_log_probs and baselines to reflect updated policy
                action_log_probs, baselines, _ = self.policy.rollout_data(obs, actions, self.sess)
                action_log_probs, baselines = action_log_probs.reshape(-1, 1), baselines.reshape(-1, 1)
            print('Optimizing meta-policy...')
            start = time.time()
            self.optimizer.train(obs, next_obs, actions, action_log_probs, returns, mb_task_inds, self.sess)
            end = time.time()
            print('    Done! ({}s)'.format(end-start))
            if iter_ % inv_save_freq == 0:
                self.saver.save(self.sess, '{}_{}'.format(self.save_path, iter_))
        self.saver.save(self.sess, '{}_{}'.format(self.save_path, iter_))

    def test_update(self, batch_timesteps=10000, max_ep_len=500):
        assert len(self.env_fns) == 1, 'should test on one task at a time'
        envs_per_task = int(np.ceil(batch_timesteps / max_ep_len))
        sampler = MetaParallelEnvExecutor(self.env_fns, envs_per_task, max_ep_len)
        obs, next_obs, actions, action_log_probs, baselines, returns, rewards \
            = collect_and_process_rollouts(sampler, self.policy, self.sess, batch_timesteps)
        self.optimizer.test(obs, next_obs, actions, action_log_probs, returns, self.sess)

class RL:
    def __init__(self,
        name,
        env_fn,
        checkpoint=None
    ):
        with tf.variable_scope(name):
            self.env_fn = env_fn
            self.ob_dim = env_fn().observation_space.shape[0]
            self.action_dim = env_fn().action_space.shape[0]

            self.policy = GaussianMLPPolicy('policy', self.ob_dim, self.action_dim, hidden_dims=[64], learn_vars=True)

            self.saver = tf.train.Saver()

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, batch_timesteps=10000, max_ep_len=500):
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, values, value_targets, advantages, rewards = collect_and_process_rollouts(self.env_fn, self.policy, self.sess, batch_timesteps, max_ep_len)
            self.policy.optimizer.train(obs, next_obs, actions, action_log_probs, values, value_targets, advantages, self.sess)
