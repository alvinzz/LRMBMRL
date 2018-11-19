import tensorflow as tf

class MLP:
    def __init__(
        self,
        name,
        in_dim,
        out_dim,
        out_activation=None,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.tanh,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
        reuse_scope=False,
    ):
        self.params = {}
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        with tf.variable_scope(name, reuse=reuse_scope):
            if self.hidden_dims:
                self.params['W0'] = tf.get_variable('W0', [in_dim, hidden_dims[0]], initializer=weight_init())
                self.params['b0'] = tf.get_variable('b0', [hidden_dims[0]], initializer=bias_init())

                for i in range(1, len(hidden_dims)):
                    self.params['W{}'.format(i)] = tf.get_variable('W{}'.format(i), [hidden_dims[i-1], hidden_dims[i]], initializer=weight_init())
                    self.params['b{}'.format(i)] = tf.get_variable('b{}'.format(i), [hidden_dims[i]], initializer=bias_init())

                i = len(hidden_dims)
                self.params['W{}'.format(i)] = tf.get_variable('W{}'.format(i), [hidden_dims[-1], out_dim], initializer=weight_init())
                self.params['b{}'.format(i)] = tf.get_variable('b{}'.format(i), [out_dim], initializer=bias_init())
            else:
                self.params['W0'] = tf.get_variable('W0', [in_dim, out_dim], initializer=weight_init())
                self.params['b0'] = tf.get_variable('b0', [out_dim], initializer=bias_init())

    def forward(self, in_tensor):
        layers = {}
        layers['in'] = in_tensor
        if self.hidden_dims:
            layers['h0'] = self.hidden_activation(tf.matmul(layers['in'], self.params['W0']) + self.params['b0'])
            for i in range(1, len(self.hidden_dims)):
                layers['h{}'.format(i)] = self.hidden_activation(
                    tf.matmul(layers['h{}'.format(i-1)], self.params['W{}'.format(i)])
                    + self.params['b{}'.format(i)]
                )
            i = len(self.hidden_dims)
            if self.out_activation is None:
                layers['out'] = tf.matmul(layers['h{}'.format(i-1)], self.params['W{}'.format(i)]) \
                    + self.params['b{}'.format(i)]
            else:
                layers['out'] = self.out_activation(
                    tf.matmul(layers['h{}'.format(i-1)], self.params['W{}'.format(i)])
                    + self.params['b{}'.format(i)]
                )
        else:
            if self.out_activation is None:
                layers['out'] = tf.matmul(layers['in'], self.params['W0']) \
                    + self.params['b0']
            else:
                layers['out'] = self.out_activation(
                    tf.matmul(layers['in'], self.params['W0'])
                    + self.params['b0']
                )
        return layers
