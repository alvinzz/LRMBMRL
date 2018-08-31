import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP
from optimizers import ClipPPO

class RewardPredictor:
    def __init__(
        self,
        name,
        latent_dim,
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

            self.rpred_network = MLP('model', latent_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.zs)
            self.pred_r = self.rpred_network.layers['out']

    def predict(self, zs, global_session):
        pred_r = global_session.run(
            self.pred_r,
            feed_dict={self.zs: zs}
        )
        return pred_r
