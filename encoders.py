import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP
from optimizers import ClipPPO

class GaussianEncoder:
    def __init__(
        self,
        name,
        ob_dim,
        latent_dim,
        out_activation=None,
        hidden_dims=[64, 64, 64],
        hidden_activation=tf.nn.tanh,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            self.mean_network = MLP('means', ob_dim, latent_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.means = self.mean_network.layers['out']

            self.log_var_network = MLP('log_vars', ob_dim, latent_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.obs)
            self.log_vars = self.log_var_network.layers['out']

            self.distribution = DiagGaussian(self.means, self.log_vars)
            self.zs = self.distribution.sample()

    def sample_encode(self, obs, global_session):
        zs = global_session.run(
            self.zs,
            feed_dict={self.obs: obs}
        )
        return zs

    def distr_encode(self, obs, global_session):
        distr = global_session.run(
            self.distribution,
            feed_dict={self.obs: obs}
        )
        return distr
