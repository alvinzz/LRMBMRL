from policies import GaussianMLPPolicy, ConvGaussianMLPPolicy
import tensorflow as tf
import numpy as np
from rollouts import *

from distributions import DiagGaussian

class MetaRL:
    def __init__(self,
        name,
        env_fns,
        expert_trajs,
        checkpoint=None,
        save_path='model',
    ):
        with tf.variable_scope(name):
            self.env_fns = env_fns
            self.n_tasks = len(self.env_fns)
            sample_env = list(env_fns.values())[0]()
            self.ob_dim = sample_env.observation_space.shape[0]
            self.action_dim = sample_env.action_space.shape[0]

            self.expert_trajs = expert_trajs

            # self.policy = GaussianMLPPolicy('policy', self.expert_trajs, self.ob_dim, self.action_dim, hidden_dims=[64], learn_vars=True)
            self.policy = ConvGaussianMLPPolicy('policy', self.expert_trajs, self.ob_dim, self.action_dim, learn_vars=True)

            self.saver = tf.train.Saver()
            self.save_path = save_path

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, n_iters, batch_timesteps=10000, max_ep_len=500, inv_save_freq=250):
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, baselines, returns, rewards = {}, {}, {}, {}, {}, {}, {}
            for task in self.env_fns.keys():
                obs[task], next_obs[task], actions[task], action_log_probs[task], \
                baselines[task], returns[task], rewards[task] \
                    = collect_and_process_rollouts(self.env_fns[task], self.policy, self.sess, batch_timesteps, max_ep_len)
            self.policy.optimizer.train(obs, next_obs, actions, action_log_probs, returns, self.sess)
            if iter_ % inv_save_freq == 0:
                self.saver.save(self.sess, save_path)
        self.saver.save(self.sess, save_path)

    def test_update(self, batch_timesteps=10000, max_ep_len=500):
        assert len(self.env_fns) == 1, 'should test on one task at a time'
        env_fn = list(self.env_fns.values())[0]
        obs, next_obs, actions, action_log_probs, baselines, returns, rewards \
            = collect_and_process_rollouts(env_fn, self.policy, self.sess, batch_timesteps, max_ep_len)
        self.policy.optimizer.test(obs, next_obs, actions, action_log_probs, returns, self.sess)

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
