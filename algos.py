from policies import GaussianMLPPolicy
import tensorflow as tf
import numpy as np
from rollouts import *

from distributions import DiagGaussian

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
