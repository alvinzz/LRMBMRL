from policies import GaussianMLPPolicy
import tensorflow as tf
import numpy as np
from rollouts import *

from encoders import *
from decoders import *
from models import *
from r_predictors import *
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

class ModelLearning:
    def __init__(
        self,
        name,
        env_fn,
        dataset,
        latent_dim=6,
        optimizer=tf.train.AdamOptimizer,
        lr=0.01,
        checkpoint=None
    ):
        with tf.variable_scope(name):
            self.env_fn = env_fn
            self.ob_dim = env_fn().observation_space.shape[0]
            self.action_dim = env_fn().action_space.shape[0]
            self.latent_dim = latent_dim

            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)

            self.obs = tf.placeholder(tf.float32, shape=[None, self.ob_dim], name='obs')
            self.next_obs = tf.placeholder(tf.float32, shape=[None, self.ob_dim], name='next_obs')
            self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name='rewards')

            self.encoder = GaussianEncoder('encoder', self.ob_dim, self.latent_dim, self.obs, hidden_dims=[10])
            self.next_encoder = GaussianEncoder('encoder', self.ob_dim, self.latent_dim, self.next_obs, hidden_dims=[10], reuse_scope=True)
            self.decoder = Decoder('decoder', self.latent_dim, self.ob_dim, self.encoder.zs, hidden_dims=[10])
            self.model = Model('model', self.latent_dim, self.action_dim, self.encoder.zs, hidden_dims=[10, 10])
            self.reward_predictor = RewardPredictor('r_pred', self.latent_dim, self.encoder.zs, hidden_dims=[10, 10])

            self._create_loss()
            self.optimizer = optimizer(lr)
            self.train_op = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

            if checkpoint:
                self.saver.restore(self.sess, checkpoint)

        self.data_obs = np.array(dataset['obs'])
        self.data_next_obs = np.array(dataset['next_obs'])
        self.data_actions = np.array(dataset['actions'])
        self.data_rewards = np.array(dataset['rewards'])

    def train(self, n_epochs=10000, mb_size=20):
        for epoch in range(n_epochs):
            mb_inds = np.random.randint(0, self.data_obs.shape[0], mb_size)
            feed_dict = {
                self.encoder.obs: self.data_obs[mb_inds],
                self.next_encoder.obs: self.data_next_obs[mb_inds],
                self.model.actions: self.data_actions[mb_inds],
                self.rewards: self.data_rewards[mb_inds],
            }
            loss_dict, _ = self.sess.run([self.loss_dict, self.train_op], feed_dict=feed_dict)
            if epoch % 1000 == 0:
                print('Loss: {}'.format(loss_dict))

    def _create_loss(self):
        std_normal = DiagGaussian(tf.zeros_like(self.encoder.zs), tf.zeros_like(self.encoder.zs))

        self.latent_prior_loss = self.encoder.distribution.kl(std_normal)
        self.reconstr_loss = tf.reduce_mean(tf.square(self.decoder.decoded - self.encoder.obs))
        self.model_loss = -self.next_encoder.distribution.log_prob(self.model.pred_z)
        self.r_pred_loss = tf.reduce_mean(tf.square(self.reward_predictor.pred_r - self.rewards))

        self.loss_dict = {
            'latent_prior': self.latent_prior_loss,
            'reconstruction': self.reconstr_loss,
            'model': self.model_loss,
            'reward_prediction': self.r_pred_loss,
        }

        self.loss = 1. * self.latent_prior_loss \
                  + 0. * self.reconstr_loss \
                  + 1. * self.model_loss \
                  + 1. * self.r_pred_loss
