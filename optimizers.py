import tensorflow as tf
import numpy as np
from utils import batchify

class MILOptimizer:
    def __init__(self,
        policy, expert_trajs, meta_batch_size,
        clip_param=0.1, max_grad_norm=0.1, min_log_var=-2,
        optimizer=tf.train.AdamOptimizer, learning_rate=0.01, optimizer_epsilon=1e-5,
        inner_learning_rate=1.0,
    ):
        # process expert trajs
        self.expert_trajs = expert_trajs
        self.tasks = sorted(self.expert_trajs.keys())
        self.n_tasks = len(self.expert_trajs)
        self.expert_obs = tf.stack([
            tf.constant(self.expert_trajs[task]['obs'], name='expert_obs', dtype=tf.float32)
            for task in self.tasks
        ])
        self.expert_actions = tf.stack([
            tf.constant(self.expert_trajs[task]['actions'], name='expert_actions', dtype=tf.float32)
            for task in self.tasks
        ])

        # get expert traj data for tasks in minibatch
        self.meta_batch_size = meta_batch_size
        self.mb_task_inds = tf.placeholder(tf.int32, self.meta_batch_size)
        self.mb_expert_obs = tf.gather_nd(self.expert_obs, tf.expand_dims(self.mb_task_inds, axis=1))
        self.mb_expert_actions = tf.gather_nd(self.expert_actions, tf.expand_dims(self.mb_task_inds, axis=1))

        # collect parameters
        self.policy = policy
        self.params = self.policy.params

        # placeholders for policy gradient
        self.returns = tf.placeholder(tf.float32, shape=[None, 1], name='returns')
        self.old_action_log_probs = tf.placeholder(tf.float32, shape=[None, 1], name='old_action_log_probs')

        # per-task dicts (for quantities needed for policy gradient)
        self.action_log_prob_task_dict = self._get_per_task_tensor_dict(tf.expand_dims(self.policy.action_log_probs, axis=1))
        self.old_action_log_prob_task_dict = self._get_per_task_tensor_dict(self.old_action_log_probs)
        self.returns_task_dict = self._get_per_task_tensor_dict(self.returns)
        self.baselines_task_dict = self._get_per_task_tensor_dict(self.policy.baselines)

        # meta-learning
        self.inner_loss_task_dict = {}
        self.inner_grads_task_dict = {}
        self.postupdate_params_task_dict = {}
        self.inner_update_op_task_dict = {}
        self.postupdate_dists_task_dict = {}
        self.il_loss_task_dict = {}
        self.grads_task_dict = {}
        self.zipped_grads_task_dict = {}
        for task in range(self.meta_batch_size):
            print('Creating computation graph for meta-batch task {}...'.format(task))
            # inner update: policy gradient surrogate loss + grads
            task_action_prob_ratio = tf.exp(self.action_log_prob_task_dict[task] - self.old_action_log_prob_task_dict[task])
            task_policy_loss = -task_action_prob_ratio \
                * (self.returns_task_dict[task] - self.baselines_task_dict[task])
            task_clipped_policy_loss = -tf.clip_by_value(task_action_prob_ratio, 1-clip_param, 1+clip_param) \
                * (self.returns_task_dict[task] - self.baselines_task_dict[task])
            self.inner_loss_task_dict[task] = tf.reduce_mean(tf.maximum(task_policy_loss, task_clipped_policy_loss))
            self.inner_grads_task_dict[task], task_zipped_inner_grads, self.postupdate_params_task_dict[task] \
                = self.collect_grads(
                    self.params, self.inner_loss_task_dict[task],
                    max_grad_norm, inner_learning_rate,
                    noupdate_keys=['conv_network']
                )
            self.inner_update_op_task_dict[task] \
                = tf.train.GradientDescentOptimizer(inner_learning_rate).apply_gradients(task_zipped_inner_grads)

            # get postupdate policy dist
            task_postupdate_means, task_postupdate_log_vars = self.policy.forward(
                self.mb_expert_obs[task],
                params=self.postupdate_params_task_dict[task]
            )
            task_postupdate_log_vars = tf.maximum(task_postupdate_log_vars, min_log_var)
            self.postupdate_dists_task_dict[task] = self.policy.distribution_class(task_postupdate_means, task_postupdate_log_vars)
            task_expert_action_log_probs = self.postupdate_dists_task_dict[task].log_prob(self.mb_expert_actions[task])

            # meta-update: behavior cloning loss + gradients for the postupdate dist
            self.il_loss_task_dict[task] = -tf.reduce_mean(task_expert_action_log_probs)
            self.grads_task_dict[task], self.zipped_grads_task_dict[task], _ = self.collect_grads(self.params, self.il_loss_task_dict[task])

        # collect losses and grads across all tasks, and optimize
        self.zipped_grads = None
        self.il_loss = tf.constant(0, dtype=tf.float32)
        for task in range(self.meta_batch_size):
            self.il_loss += self.il_loss_task_dict[task]
            if self.zipped_grads is None:
                self.zipped_grads = self.zipped_grads_task_dict[task]
            else:
                self.zipped_grads = [
                    (
                        self.zipped_grads[i][0] + self.zipped_grads_task_dict[task][i][0],
                        self.zipped_grads[i][1]
                    )
                    for i in range(len(self.zipped_grads))
                ]
        self.zipped_grads = [
            (
                self.zipped_grads[i][0] / len(self.tasks),
                self.zipped_grads[i][1]
            )
            for i in range(len(self.zipped_grads))
        ]
        self.il_loss /= len(self.tasks)
        self.optimizer = optimizer(learning_rate=learning_rate, epsilon=optimizer_epsilon)
        self.train_op = self.optimizer.apply_gradients(self.zipped_grads)

    def train(self,
        obs, next_obs, actions, action_log_probs, returns, mb_task_inds,
        global_session,
    ):
        il_loss, _ = global_session.run(
            [self.il_loss, self.train_op],
            feed_dict={
                self.policy.obs: obs,
                self.policy.actions: actions,
                self.old_action_log_probs: action_log_probs,
                self.returns: returns,
                self.mb_task_inds: mb_task_inds
            }
        )
        print('IL loss:', il_loss)

    def test(self,
        obs, next_obs, actions, action_log_probs, returns,
        global_session,
    ):
        global_session.run(
            self.inner_update_op_task_dict[0],
            feed_dict={
                self.policy.obs: obs,
                self.policy.actions: actions,
                self.old_action_log_probs: action_log_probs,
                self.returns: returns,
                self.mb_task_inds: [0],
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

    def _get_per_task_tensor_dict(self, tensor):
        if tensor.shape[0] is not None:
            per_task_tensor_dict = {task: tensor for task in range(self.meta_batch_size)}
        else:
            per_task_tensor = tf.unstack(
                tf.transpose(
                    tf.reshape(tensor, (-1, self.n_tasks, tensor.shape[1].value)),
                    [1, 0, 2],
                ),
                num=self.n_tasks,
                axis=0,
            )
            per_task_tensor_dict = {task: tensor for (task, tensor) in zip(range(self.meta_batch_size), per_task_tensor)}
        return per_task_tensor_dict

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
