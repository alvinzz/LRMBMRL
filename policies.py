import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP
from optimizers import ClipPPO, MILOptimizer

class GaussianMLPPolicy:
    def __init__(
        self,
        name,
        expert_trajs,
        ob_dim,
        action_dim,
        learn_vars=True,
        var_network=False, # NN if true, else trainable params indep of obs
        out_activation=None,
        hidden_dims=[64, 64, 64],
        hidden_activation=tf.nn.leaky_relu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        optimizer=MILOptimizer,
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            # policy net
            self.mean_network = MLP('means', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init)
            self.means = self.mean_network.forward(self.obs)['out']

            if learn_vars:
                if var_network:
                    self.log_var_network = MLP('log_vars', ob_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init)
                    self.log_vars = self.log_var_network.forward(self.obs)['out']
                else:
                    self.log_vars = tf.get_variable('log_vars', trainable=True, initializer=-np.ones((1, action_dim), dtype=np.float32))
            else:
                self.log_vars = tf.get_variable('log_vars', trainable=False, initializer=np.zeros((1, action_dim), dtype=np.float32))

            self.distribution = DiagGaussian(self.means, self.log_vars)
            self.sampled_actions = self.distribution.sample()

            self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
            self.action_log_probs = self.distribution.log_prob(self.actions)
            self.entropies = self.distribution.entropy()

            # baseline net
            # self.baseline_network = MLP('baseline', ob_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init)
            # self.baseline_network = MLP('baseline', ob_dim, 1, out_activation=out_activation, hidden_dims=[], hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init)
            # self.baselines = self.baseline_network.forward(self.obs)['out']
            self.baselines = tf.get_variable('baseline', trainable=True, initializer=np.zeros((1, 1), dtype=np.float32))

            # training, MIL
            self.optimizer = optimizer(ob_dim, action_dim, self, expert_trajs)

    def act(self, obs, global_session):
        actions = global_session.run(
            self.sampled_actions,
            feed_dict={self.obs: obs}
        )
        return actions

    def rollout_data(self, obs, actions, global_session):
        action_log_probs, baselines, entropies = global_session.run(
            [self.action_log_probs, self.baselines, self.entropies],
            feed_dict={self.obs: obs, self.actions: actions}
        )
        return action_log_probs, baselines, entropies
