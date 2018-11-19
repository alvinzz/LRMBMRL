import tensorflow as tf
import numpy as np
from utils import batchify
from distributions import DiagGaussian

class MILOptimizer:
    def __init__(self,
        ob_dim, action_dim, policy, expert_trajs,
        clip_param=0.1, max_grad_norm=0.1,
        optimizer=tf.train.AdamOptimizer, learning_rate=0.01, optimizer_epsilon=1e-5,
        inner_learning_rate=1.0,
    ):
        self.expert_trajs = expert_trajs
        self.optimizer = optimizer(learning_rate=learning_rate, epsilon=optimizer_epsilon)

        self.returns = tf.placeholder(tf.float32, shape=[None, 1], name='returns')
        self.old_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_log_probs')

        self.policy = policy
        self.obs = self.policy.obs
        self.distribution = self.policy.distribution
        self.actions = self.policy.actions
        self.action_log_probs = self.policy.action_log_probs
        self.baselines = self.policy.baselines
        self.log_vars = self.policy.log_vars

        # collect policy parameter dict
        self.params = {}
        for (k, v) in self.policy.mean_network.params.items():
            self.params['mean_network/' + k] = v
        if hasattr(self.policy, 'baseline_network'):
            for (k, v) in self.policy.baseline_network.params.items():
                self.params['baseline_network/' + k] = v
        elif self.policy.baselines.trainable:
            self.params['baseline'] = self.policy.baselines
        if hasattr(self.policy, 'log_var_network'):
            for (k, v) in self.policy.log_var_network.params.items():
                self.params['var_network/' + k] = v
        elif self.policy.log_vars.trainable:
            self.params['var'] = self.policy.log_vars

        # policy gradient surrogate loss
        self.action_prob_ratio = tf.exp(tf.expand_dims(self.action_log_probs, axis=1) - self.old_action_log_probs)
        self.policy_loss = -self.action_prob_ratio * (self.returns - self.baselines)
        self.clipped_policy_loss = -tf.clip_by_value(self.action_prob_ratio, 1-clip_param, 1+clip_param) \
            * (self.returns - self.baselines)
        self.surr_policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss, self.clipped_policy_loss))
        self.rl_loss = self.surr_policy_loss

        # inner gradients
        self.inner_grads = {}
        for (k, v) in self.params.items():
            self.inner_grads[k], _ = tf.clip_by_global_norm(
                tf.gradients(self.surr_policy_loss, self.params[k]),
                max_grad_norm,
            )
            self.inner_grads[k] = self.inner_grads[k][0]
        zipped_inner_grads = []
        for k in self.params.keys():
            zipped_inner_grads.append((self.inner_grads[k], self.params[k]))
        self.inner_update_op = tf.train.GradientDescentOptimizer(inner_learning_rate).apply_gradients(zipped_inner_grads)

        # apply inner gradient updates
        self.postupdate_params = {}
        for (k, v) in self.params.items():
            self.postupdate_params[k] = self.params[k] - inner_learning_rate*self.inner_grads[k]
        for k in self.policy.mean_network.params.keys():
            self.policy.mean_network.params[k] = self.postupdate_params['mean_network/' + k]
        if hasattr(self.policy, 'log_var_network'):
            for k in self.policy.log_var_network.params.keys():
                self.policy.log_var_network.params[k] = self.postupdate_params['var_network/' + k]
        elif self.policy.log_vars.trainable:
            self.policy.log_var = self.postupdate_params['var']

        ### meta-rl
        # get postupdate policy dist
        self.expert_obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='expert_obs')
        self.expert_actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
        self.postupdate_means = self.policy.mean_network.forward(self.expert_obs)['out']
        if hasattr(self.policy, 'log_var_network'):
            self.postupdate_vars = self.policy.log_var_network.forward(self.expert_obs)['out']
        else:
            self.postupdate_vars = self.policy.log_var
        self.postupdate_dist = DiagGaussian(self.postupdate_means, self.postupdate_vars)
        self.expert_action_log_probs = self.postupdate_dist.log_prob(self.expert_actions)
        # behavior cloning loss + gradients for postupdate dist
        self.il_loss = -tf.reduce_mean(self.expert_action_log_probs)
        self.grads = {}
        for (k, v) in self.params.items():
            self.grads[k] = tf.gradients(self.il_loss, v)[0]
        zipped_grads = []
        for k in self.params.keys():
            zipped_grads.append((self.grads[k], self.params[k]))
        self.train_op = self.optimizer.apply_gradients(zipped_grads)

    def train(self,
        obs, next_obs, actions, action_log_probs, returns,
        global_session,
    ):
        grads = {}
        il_loss = 0
        for task in self.expert_trajs.keys():
            data = [obs[task], actions[task], action_log_probs[task], returns[task]]
            # todo?: minibatches
            mb_obs, mb_actions, mb_action_log_probs, mb_returns = data
            task_grads, task_il_loss = global_session.run(
                [self.grads, self.il_loss],
                feed_dict={
                    self.obs: mb_obs,
                    self.actions: mb_actions,
                    self.old_action_log_probs: mb_action_log_probs,
                    self.returns: mb_returns,
                    self.expert_obs: self.expert_trajs[task]['obs'],
                    self.expert_actions: self.expert_trajs[task]['actions'],
                }
            )
            for k in task_grads.keys():
                if k not in grads.keys():
                    grads[k] = task_grads[k]
                else:
                    grads[k] += task_grads[k]
            il_loss += task_il_loss
        for k in grads.keys():
            grads[k] /= len(self.expert_trajs)
        il_loss /= len(self.expert_trajs)
        print('IL loss:', il_loss)
        global_session.run(
            self.train_op,
            feed_dict={
                self.grads[k]: grads[k]
                for k in self.grads.keys()
            }
        )

    def test(self,
        obs, next_obs, actions, action_log_probs, returns,
        global_session,
    ):
        global_session.run(
            self.inner_update_op,
            feed_dict={
                self.obs: obs,
                self.actions: actions,
                self.old_action_log_probs: action_log_probs,
                self.returns: returns,
            }
        )


class ClipPPO:
    def __init__(self,
        ob_dim, action_dim, policy,
        clip_param=0.1, max_grad_norm=0.1,
        optimizer=tf.train.AdamOptimizer, learning_rate=1e-4, optimizer_epsilon=1e-5
    ):
        self.optimizer = optimizer(learning_rate=learning_rate, epsilon=optimizer_epsilon)

        self.old_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_log_probs')
        self.old_values = tf.placeholder(tf.float32, shape=[None, 1], name='old_values')
        self.value_targets = tf.placeholder(tf.float32, shape=[None, 1], name='value_targets')
        self.advantages = tf.placeholder(tf.float32, shape=[None, 1], name='advantages')

        self.policy = policy
        self.obs = self.policy.obs
        self.distribution = self.policy.distribution
        self.actions = self.policy.actions
        self.action_log_probs = self.policy.action_log_probs
        self.values = self.policy.values
        self.log_vars = self.policy.log_vars

        # clipped policy loss
        self.action_prob_ratio = tf.exp(tf.expand_dims(self.action_log_probs, axis=1) - self.old_action_log_probs)
        self.policy_loss = -self.action_prob_ratio * self.advantages
        self.clipped_policy_loss = -tf.clip_by_value(self.action_prob_ratio, 1-clip_param, 1+clip_param) * self.advantages
        self.surr_policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss, self.clipped_policy_loss))

        # value loss
        self.value_loss = tf.square(self.value_targets - self.values)
        self.clipped_value_loss = tf.square(
            self.value_targets - (self.old_values + tf.clip_by_value(self.values - self.old_values, -clip_param, clip_param))
        )
        # self.surr_value_loss = 0.5 * tf.reduce_mean(tf.maximum(self.value_loss, self.clipped_value_loss))
        self.surr_value_loss = 0.5 * tf.reduce_mean(self.value_loss)

        # total loss
        self.loss = self.surr_policy_loss + self.surr_value_loss

        # gradients
        self.params = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.params)
        self.grads, _ = tf.clip_by_global_norm(self.grads, max_grad_norm)
        self.grads = list(zip(self.grads, self.params))
        self.train_op = self.optimizer.apply_gradients(self.grads)

    def train(self,
        obs, next_obs, actions, action_log_probs, values, value_targets, advantages,
        global_session,
        n_iters=10, batch_size=64
    ):
        data = [obs, actions, action_log_probs, values, value_targets, advantages]
        for iter_ in range(n_iters):
            batched_data = batchify(data, batch_size)
            for minibatch in batched_data:
                mb_obs, mb_actions, mb_action_log_probs, mb_values, mb_value_targets, mb_advantages = minibatch
                # normalize advantages here?
                # mb_advantages = (mb_advantages - np.mean(mb_advantages)) / (np.std(mb_advantages) + 1e-8)
                global_session.run(
                    self.train_op,
                    feed_dict={self.obs: mb_obs, self.actions: mb_actions, self.old_action_log_probs: mb_action_log_probs, self.old_values: mb_values, self.value_targets: mb_value_targets, self.advantages: mb_advantages}
                )
