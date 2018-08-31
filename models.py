import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP
from optimizers import ClipPPO

class Model:
    def __init__(
        self,
        name,
        latent_dim,
        action_dim,
        zs,
        out_activation=None,
        hidden_dims=[64, 64, 64],
        hidden_activation=tf.nn.tanh,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer
    ):
        with tf.variable_scope(name):
            # self.zs = tf.placeholder(tf.float32, shape=[None, latent_dim], name='zs')
            self.zs = zs
            self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
            self.za_concat = tf.concat([self.zs, self.actions], axis=1)

            self.model_network = MLP('model', latent_dim+action_dim, latent_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.za_concat)
            self.pred_z = self.model_network.layers['out']

    def predict(self, zs, actions, global_session):
        pred_z = global_session.run(
            self.pred_z,
            feed_dict={self.zs: zs, self.actions: actions}
        )
        return pred_z
