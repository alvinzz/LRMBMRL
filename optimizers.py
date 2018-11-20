import tensorflow as tf
import numpy as np
from utils import batchify

class MILOptimizer:
    def __init__(self,
        ob_dim, action_dim, policy, expert_trajs,
        clip_param=0.1, max_grad_norm=0.1, min_log_var=-2,
        optimizer=tf.train.AdamOptimizer, learning_rate=0.01, optimizer_epsilon=1e-5,
        inner_learning_rate=1.0,
    ):
        self.expert_trajs = expert_trajs
        self.optimizer = optimizer(learning_rate=learning_rate, epsilon=optimizer_epsilon)

        self.returns = tf.placeholder(tf.float32, shape=[None, 1], name='returns')
        self.old_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_log_probs')

        self.policy = policy
        self.params = self.policy.params
        self.obs = self.policy.obs
        self.distribution = self.policy.distribution
        self.actions = self.policy.actions
        self.action_log_probs = self.policy.action_log_probs
        self.baselines = self.policy.baselines
        self.log_vars = self.policy.log_vars

        # inner update: policy gradient surrogate loss + grads
        self.action_prob_ratio = tf.exp(tf.expand_dims(self.action_log_probs, axis=1) - self.old_action_log_probs)
        self.policy_loss = -self.action_prob_ratio * (self.returns - self.baselines)
        self.clipped_policy_loss = -tf.clip_by_value(self.action_prob_ratio, 1-clip_param, 1+clip_param) \
            * (self.returns - self.baselines)
        self.surr_policy_loss = tf.reduce_mean(tf.maximum(self.policy_loss, self.clipped_policy_loss))
        self.inner_loss = self.surr_policy_loss
        self.inner_grads, self.zipped_inner_grads, self.postupdate_params = self.collect_grads(
            self.params, self.inner_loss, max_grad_norm, inner_learning_rate, noupdate_keys=['conv_network'],
        )
        self.inner_update_op = tf.train.GradientDescentOptimizer(inner_learning_rate).apply_gradients(self.zipped_inner_grads)

        # get postupdate policy dist
        self.expert_obs = tf.placeholder(tf.float32, shape=[None, ob_dim], name='expert_obs')
        self.expert_actions = tf.placeholder(tf.float32, shape=[None, action_dim], name='actions')
        self.postupdate_means, self.postupdate_log_vars = self.policy.forward(self.expert_obs, params=self.postupdate_params)
        self.postupdate_log_vars = tf.maximum(self.postupdate_log_vars, min_log_var)
        self.postupdate_dist = self.policy.distribution_class(self.postupdate_means, self.postupdate_log_vars)
        self.expert_action_log_probs = self.postupdate_dist.log_prob(self.expert_actions)

        # meta-update: behavior cloning loss + gradients for the postupdate dist
        self.il_loss = -tf.reduce_mean(self.expert_action_log_probs)
        self.grads, self.zipped_grads, _ = self.collect_grads(self.params, self.il_loss)
        self.train_op = self.optimizer.apply_gradients(self.zipped_grads)

    def train(self,
        obs, next_obs, actions, action_log_probs, returns,
        global_session,
    ):
        il_loss = 0
        for (i, task) in enumerate(self.expert_trajs.keys()):
            data = [obs[task], actions[task], action_log_probs[task], returns[task]]
            # todo?: minibatches
            mb_obs, mb_actions, mb_action_log_probs, mb_returns = data
            task_grads, task_il_loss = global_session.run(
                [self.zipped_grads, self.il_loss],
                feed_dict={
                    self.obs: mb_obs,
                    self.actions: mb_actions,
                    self.old_action_log_probs: mb_action_log_probs,
                    self.returns: mb_returns,
                    self.expert_obs: self.expert_trajs[task]['obs'],
                    self.expert_actions: self.expert_trajs[task]['actions'],
                }
            )
            if i == 0:
                grads = task_grads
            else:
                grads = [(grads[i][0] + task_grads[i][0], grads[i][1]) for i in range(len(grads))]
            il_loss += task_il_loss
        grads = [(grads[i][0] / len(self.expert_trajs), grads[i][1]) for i in range(len(grads))]
        il_loss /= len(self.expert_trajs)
        print('IL loss:', il_loss)
        global_session.run(
            self.train_op,
            feed_dict={
                self.zipped_grads[i][0]: grads[i][0]
                for i in range(len(grads))
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

    def collect_grads(self, params, loss, max_grad_norm=0.1, learning_rate=0.1, noupdate_keys=['conv_network']):
        grads, zipped_grads, postupdate_params = {}, [], {}
        for (k, v) in params.items():
            if type(v) != dict: # is tf variable reference
                grads[k], _ = tf.clip_by_global_norm(
                    tf.gradients(loss, params[k]),
                    max_grad_norm,
                )
                grads[k] = grads[k][0]
                if k in noupdate_keys:
                    grads[k] = tf.zeros_like(grads[k])
                postupdate_params[k] = params[k] - learning_rate*grads[k]
                zipped_grads.append((grads[k], params[k]))
            else:
                grads[k], sub_zipped_grads, postupdate_params[k] = self.collect_grads(params[k], loss)
                zipped_grads.extend(sub_zipped_grads)
        return grads, zipped_grads, postupdate_params

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
