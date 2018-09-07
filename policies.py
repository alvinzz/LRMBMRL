import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP
from optimizers import ClipPPO

class GaussianMLPPolicy:
    def __init__(
        self,
        name,
        ob_dim,
        action_dim,
        n_tasks=15,
        task_latent_dim=2,
        learn_vars=True,
        var_network=False, # NN if true, else trainable params indep of obs
        out_activation=None,
        hidden_dims=[64, 64, 64],
        hidden_activation=tf.nn.tanh,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        optimizer=ClipPPO
    ):
        with tf.variable_scope(name):
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            initial_latents = np.concatenate(
                (
                    np.random.normal(size=(n_tasks, task_latent_dim)),
                    -25*np.ones((n_tasks, task_latent_dim))
                ),
                axis=1
            ).astype(np.float32)
            self.task_latents = tf.get_variable('task_latents', initializer=initial_latents, dtype=np.float32)
            self.task_ids = tf.placeholder(tf.int32, shape=[None, 1], name='task_ids')
            self.task_latent_distribution = DiagGaussian(
                tf.gather_nd(self.task_latents, self.task_ids)[:, :task_latent_dim],
                tf.gather_nd(self.task_latents, self.task_ids)[:, task_latent_dim:]
            )
            self.zs = self.task_latent_distribution.sample()

            self.policy_input = tf.concat((self.obs, self.zs), axis=1)
            self.debug = self.policy_input

            # policy net
            self.mean_network = MLP('means', ob_dim+task_latent_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.policy_input)
            self.means = self.mean_network.layers['out']

            if learn_vars:
                if var_network:
                    self.log_var_network = MLP('log_vars', ob_dim+task_latent_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.policy_input)
                    self.log_vars = self.log_var_network.layers['out']
                else:
                    self.log_vars = tf.get_variable('log_vars', trainable=True, initializer=0*np.ones((1, action_dim), dtype=np.float32))
            else:
                self.log_vars = tf.get_variable('log_vars', trainable=False, initializer=0*np.ones((1, action_dim), dtype=np.float32))

            self.distribution = DiagGaussian(self.means, self.log_vars)
            self.sampled_actions = self.distribution.sample()

            self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
            self.action_log_probs = self.distribution.log_prob(self.actions)
            self.entropies = self.distribution.entropy()

            # value net
            self.value_network = MLP('values', ob_dim+task_latent_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=weight_init, bias_init=bias_init, in_layer=self.policy_input)
            self.values = self.value_network.layers['out']

            # training, PPO for now
            self.optimizer = optimizer(ob_dim, action_dim, self)

    def act(self, obs, task_ids, global_session):
        actions = global_session.run(
            self.sampled_actions,
            feed_dict={self.obs: obs, self.task_ids: task_ids}
        )
        return actions

    def rollout_data(self, obs, actions, task_ids, global_session):
        action_log_probs, values, entropies = global_session.run(
            [self.action_log_probs, self.values, self.entropies],
            feed_dict={self.obs: obs, self.actions: actions, self.task_ids: task_ids}
        )
        return action_log_probs, values, entropies

    def get_task_latents(self, global_session):
        task_latents = global_session.run(
            self.task_latents
        )
        return task_latents

    def get_debug(self, obs, task_ids, global_session):
        debug = global_session.run(
            self.debug,
            feed_dict={self.obs: obs, self.task_ids: task_ids}
        )
        return debug
