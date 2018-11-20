import tensorflow as tf

class ConvNet:
    def __init__(
        self,
        name,
        filter_sizes,
        strides,
        dilations,
        paddings,
        activation,
        out_activation,
        weight_init,
    ):
        self.params = {}
        self.filter_sizes = filter_sizes
        self.strides = [[1, stride, stride, 1] for stride in strides]
        self.dilations = [[1, dilation, dilation, 1] for dilation in dilations]
        self.paddings = paddings
        self.activation = activation
        self.out_activation = out_activation
        with tf.variable_scope(name):
            for i in range(len(self.filter_sizes)):
                self.params['filter{}'.format(i)] = tf.get_variable('filter{}'.format(i), filter_sizes[i], initializer=weight_init())

    def forward(self, input_tensor, params=None):
        if params is not None:
            orig_params = self.params
            self.params = params
        layers = {}
        layers['in'] = input_tensor
        if len(self.filter_sizes) > 1:
            layers['conv0'] = self.activation(
                tf.nn.conv2d(
                    layers['in'], self.params['filter0'],
                    self.strides[0], self.paddings[0],
                    dilations=self.dilations[0]
                )
            )
            for i in range(1, len(self.filter_sizes)-1):
                layers['conv{}'.format(i)] = self.activation(
                    tf.nn.conv2d(
                        layers['conv{}'.format(i-1)], self.params['filter{}'.format(i)],
                        self.strides[i], self.paddings[i],
                        dilations=self.dilations[i],
                    )
                )
            i = len(self.filter_sizes) - 1
            layers['out'] = self.out_activation(
                tf.nn.conv2d(
                    layers['conv{}'.format(i-1)], self.params['filter{}'.format(i)],
                    self.strides[i], self.paddings[i],
                    dilations=self.dilations[i]
                )
            )
        else:
            layers['out'] = self.out_activation(
                tf.nn.conv2d(
                    layers['in'], self.params['filter0'],
                    self.strides[0], self.paddings[0],
                    dilations=self.dilations[0]
                )
            )
        if params is not None:
            self.params = orig_params
        return layers

class MLP:
    def __init__(
        self,
        name,
        in_dim,
        out_dim,
        out_activation=tf.identity,
        hidden_dims=[64, 64],
        hidden_activation=tf.nn.leaky_relu,
        weight_init=tf.contrib.layers.xavier_initializer,
        bias_init=tf.zeros_initializer,
    ):
        self.params = {}
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
        with tf.variable_scope(name):
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

    def forward(self, input_tensor, params=None):
        if params is not None:
            orig_params = self.params
            self.params = params
        layers = {}
        layers['in'] = input_tensor
        if self.hidden_dims:
            layers['h0'] = self.hidden_activation(tf.matmul(layers['in'], self.params['W0']) + self.params['b0'])
            for i in range(1, len(self.hidden_dims)):
                layers['h{}'.format(i)] = self.hidden_activation(
                    tf.matmul(layers['h{}'.format(i-1)], self.params['W{}'.format(i)])
                    + self.params['b{}'.format(i)]
                )
            i = len(self.hidden_dims)
            layers['out'] = self.out_activation(
                tf.matmul(layers['h{}'.format(i-1)], self.params['W{}'.format(i)])
                + self.params['b{}'.format(i)]
            )
        else:
            layers['out'] = self.out_activation(
                tf.matmul(layers['in'], self.params['W0'])
                + self.params['b0']
            )
        if params is not None:
            self.params = orig_params
        return layers
