from policies import GaussianMLPPolicy
import tensorflow as tf
import numpy as np
from rollouts import *
from distributions import DiagGaussian
import pickle
import os

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

            self.policy = GaussianMLPPolicy('policy', self.ob_dim, self.action_dim, hidden_dims=[64], learn_vars=True, hidden_activation=tf.nn.relu)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

            self.name = name
            self.saver = tf.train.Saver()
            self.use_checkpoint = bool(checkpoint)
            if self.use_checkpoint:
                self.saver.restore(self.sess, checkpoint)

    def train(self, save_dir, n_iters, batch_timesteps=10000, max_ep_len=500):
        for iter_ in range(n_iters):
            print('______________')
            print('Iteration', iter_)
            obs, next_obs, actions, action_log_probs, values, value_targets, \
                advantages, rewards, avg_ep_reward, task_ids \
                = collect_and_process_rollouts(
                    self.env_fn, self.policy, self.sess, batch_timesteps, max_ep_len
                )
            actions_prob_ratio, grads, loss_dict \
                = self.policy.optimizer.train(
                    obs, next_obs, actions, action_log_probs, values,
                    value_targets, advantages, task_ids, self.sess
                )

            # keep records
            data_dict = {
                'iter': iter_,
                'avg_ep_reward': avg_ep_reward,
                # 'obs': obs,
                # 'next_obs': next_obs,
                # 'actions': actions,
                # 'action_log_probs': action_log_probs,
                # 'values': values,
                # 'value_targets': value_targets,
                # 'advantages': advantages,
                # 'rewards': rewards,
                # 'task_ids': task_ids,
                # 'actions_prob_ratio': actions_prob_ratio,
                # 'grads': grads
            }
            data_dict.update(loss_dict)
            if not self.use_checkpoint or not os.path.exists('{}/{}_training_data.pkl'.format(save_dir, self.name)):
                pickle.dump(data_dict, open('{}/{}_training_data.pkl'.format(save_dir, self.name), 'wb'))
            else:
                pickle.dump(data_dict, open('{}/{}_training_data.pkl'.format(save_dir, self.name), 'ab'))

            # save model
            if iter_ % 100 == 0 or iter_ == n_iters-1:
                self.saver.save(self.sess, '{}/{}_model'.format(save_dir, self.name))
