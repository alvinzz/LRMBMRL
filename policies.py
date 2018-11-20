import tensorflow as tf
import numpy as np
from distributions import DiagGaussian
from networks import MLP, ConvNet
from optimizers import ClipPPO, MILOptimizer
from swish import swish
from spatial_softmax import spatial_softmax

class ConvGaussianMLPPolicy:
    def __init__(
        self,
        name,
        expert_trajs,
        ob_dim,
        action_dim,

        learn_vars=True,
        var_network=False, # NN if true, else trainable params indep of obs

        hidden_dims=[100, 100, 100],
        hidden_activation=tf.nn.leaky_relu,
        out_activation=tf.identity,
        fc_weight_init=tf.contrib.layers.xavier_initializer,
        fc_bias_init=tf.zeros_initializer,

        image_size=[64, 64, 3],
        conv_filter_sizes=[
            [5, 5, 3, 20],
        ],
        conv_strides=[1],
        conv_dilations=[1],
        conv_paddings=['SAME'],
        conv_activation=swish,
        conv_out_activation=spatial_softmax,
        conv_weight_init=tf.contrib.layers.xavier_initializer,

        optimizer=MILOptimizer,
    ):
        with tf.variable_scope(name):
            self.image_size = image_size
            self.obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='obs')

            # conv net
            self.im = tf.reshape(
                self.obs[:, :np.prod(self.image_size)],
                [-1, self.image_size[0], self.image_size[1], self.image_size[2]]
            )
            self.conv_network = ConvNet('conv', conv_filter_sizes, conv_strides, conv_dilations, conv_paddings, conv_activation, conv_out_activation, weight_init=conv_weight_init)
            self.im_feats = self.conv_network.forward(self.im)['out']

            # get state
            self.aux_state = self.obs[:, np.prod(self.image_size):]
            self.state = tf.concat((self.im_feats, self.aux_state), axis=1)
            self.state_dim = self.state.get_shape()[1].value

            # policy net
            self.mean_network = MLP('means', self.state_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=fc_weight_init, bias_init=fc_bias_init)
            self.means = self.mean_network.forward(self.state)['out']

            if learn_vars:
                if var_network:
                    self.log_var_network = MLP('log_vars', self.state_dim, action_dim, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=fc_weight_init, bias_init=fc_bias_init)
                    self.log_vars = self.log_var_network.forward(self.state)['out']
                else:
                    self.log_vars = tf.get_variable('log_vars', trainable=True, initializer=-np.ones((1, action_dim), dtype=np.float32))
            else:
                self.log_vars = tf.get_variable('log_vars', trainable=False, initializer=np.zeros((1, action_dim), dtype=np.float32))

            self.distribution_class = DiagGaussian
            self.distribution = DiagGaussian(self.means, self.log_vars)
            self.sampled_actions = self.distribution.sample()

            self.actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
            self.action_log_probs = self.distribution.log_prob(self.actions)
            self.entropies = self.distribution.entropy()

            # baseline net
            # self.baseline_network = MLP('baseline', self.state_dim, 1, out_activation=out_activation, hidden_dims=hidden_dims, hidden_activation=hidden_activation, weight_init=fc_weight_init, bias_init=fc_bias_init)
            # self.baseline_network = MLP('baseline', self.state_dim, 1, out_activation=out_activation, hidden_dims=[], hidden_activation=hidden_activation, weight_init=fc_weight_init, bias_init=fc_bias_init)
            # self.baselines = self.baseline_network.forward(self.state)['out']
            self.baselines = tf.get_variable('baseline', trainable=True, initializer=np.zeros((1, 1), dtype=np.float32))

            # collect parameters + optimize
            self.collect_params()
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

    def collect_params(self):
        self.params = {}
        # conv
        self.params['conv_network'] = {}
        for (k, v) in self.conv_network.params.items():
            self.params['conv_network'][k] = v
        # mean
        self.params['mean_network'] = {}
        for (k, v) in self.mean_network.params.items():
            self.params['mean_network'][k] = v
        # baseline
        if hasattr(self, 'baseline_network'):
            self.params['baseline_network'] = {}
            for (k, v) in self.baseline_network.params.items():
                self.params['baseline_network'][k] = v
        elif self.baselines.trainable:
            self.params['baseline'] = self.baselines
        # log var
        if hasattr(self, 'log_var_network'):
            self.params['log_var_network'] = {}
            for (k, v) in self.log_var_network.params.items():
                self.params['log_var_network'][k] = v
        elif self.log_vars.trainable:
            self.params['log_var'] = self.log_vars

    def forward(self, input_tensor, params):
        im = tf.reshape(
            input_tensor[:, :np.prod(self.image_size)],
            [-1, self.image_size[0], self.image_size[1], self.image_size[2]]
        )
        im_feats = self.conv_network.forward(
            im,
            params=params['conv_network']
        )['out']
        aux_state = input_tensor[:, np.prod(self.image_size):]
        state = tf.concat((im_feats, aux_state), axis=1)
        means = self.mean_network.forward(
            state,
            params=params['mean_network']
        )['out']
        if hasattr(self, 'log_var_network'):
            log_vars = self.log_var_network.forward(
                state,
                params=params['log_var_network']
            )['out']
        elif self.log_vars.trainable:
            log_vars = params['log_var']
        else:
            log_vars = self.log_vars
        return means, log_vars

class GaussianMLPPolicy:
    def __init__(
        self,
        name,
        expert_trajs,
        ob_dim,
        action_dim,

        learn_vars=True,
        var_network=False, # NN if true, else trainable params indep of obs

        hidden_dims=[64, 64, 64],
        hidden_activation=tf.nn.leaky_relu,
        out_activation=tf.identity,
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

            self.distribution_class = DiagGaussian
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

            # collect parameters + optimize
            self.collect_params()
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

    def collect_params(self):
        self.params = {}
        # mean
        self.params['mean_network'] = {}
        for (k, v) in self.mean_network.params.items():
            self.params['mean_network'][k] = v
        # baseline
        if hasattr(self, 'baseline_network'):
            self.params['baseline_network'] = {}
            for (k, v) in self.baseline_network.params.items():
                self.params['baseline_network'][k] = v
        elif self.baselines.trainable:
            self.params['baseline'] = self.baselines
        # log var
        if hasattr(self, 'log_var_network'):
            self.params['log_var_network'] = {}
            for (k, v) in self.log_var_network.params.items():
                self.params['log_var_network'][k] = v
        elif self.log_vars.trainable:
            self.params['log_var'] = self.log_vars
        # conv
        if hasattr(self, 'conv_network'):
            self.params['conv_network'] = {}
            for (k, v) in self.conv_network.params.items():
                self.params['conv_network'][k] = v

    def forward(self, input_tensor, params):
        means = self.mean_network.forward(
            input_tensor,
            params=params['mean_network']
        )['out']
        if hasattr(self, 'log_var_network'):
            log_vars = self.log_var_network.forward(
                state,
                params=params['log_var_network']
            )['out']
        elif self.log_vars.trainable:
            log_vars = params['log_var']
        else:
            log_vars = self.log_vars
        return means, log_vars
