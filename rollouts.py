import numpy as np

def collect_and_process_rollouts(
    env_fn, policy, global_session,
    n_timesteps=10000, max_ep_len=500,
    discount=0.99, gae_lambda=0.95,
    entropy_weight=0.
):
    # collect n_timesteps of data from n_envs rollouts in parallel
    obs, next_obs, actions, env_rewards = [], [], [], []
    ep_lens = []

    n_envs = int(np.ceil(n_timesteps / max_ep_len))
    env_vec = [env_fn() for n in range(n_envs)]
    env_timesteps = [0 for n in range(n_envs)]
    obs_vec = [[env_vec[n].reset()] for n in range(n_envs)]
    next_obs_vec = [[] for n in range(n_envs)]
    actions_vec = [[] for n in range(n_envs)]
    env_rewards_vec = [[] for n in range(n_envs)]

    while len(obs) < n_timesteps:
        cur_obs = np.array([obs[-1] for obs in obs_vec])
        action_vec = policy.act(cur_obs, global_session)
        for n in range(n_envs):
            action = action_vec[n]
            # threshold actions
            threshholded_action = np.clip(action, env_vec[n].action_space.low, env_vec[n].action_space.high)
            ob, env_reward, done, info = env_vec[n].step(threshholded_action)
            obs_vec[n].append(ob)
            next_obs_vec[n].append(ob)
            actions_vec[n].append(action)
            env_rewards_vec[n].append([env_reward])
            env_timesteps[n] += 1
            if done or env_timesteps[n] >= max_ep_len:
                # record data
                obs_vec[n].pop()
                ep_lens.append(len(obs_vec[n]))
                obs.extend(obs_vec[n])
                next_obs.extend(next_obs_vec[n])
                actions.extend(actions_vec[n])
                env_rewards.extend(env_rewards_vec[n])
                # reset env
                obs_vec[n] = [env_vec[n].reset()]
                next_obs_vec[n] = []
                actions_vec[n] = []
                env_rewards_vec[n] = []
                env_timesteps[n] = 0

    obs, next_obs, actions, env_rewards = np.array(obs), np.array(next_obs), np.array(actions), np.array(env_rewards)

    # get action_probs, baselines, entropies for all timesteps
    action_log_probs, baselines, entropies = policy.rollout_data(obs, actions, global_session)
    action_log_probs, baselines, entropies = np.expand_dims(action_log_probs, axis=1), np.array(baselines), np.expand_dims(entropies, axis=1)

    # apply reward function
    rewards = env_rewards + entropy_weight * entropies
    # print('avg_ep_reward:', sum(rewards) / len(ep_lens))

    # get returns
    returns = []
    start_ind = 0
    for ep_len in ep_lens:
        ep_rewards = rewards[start_ind : start_ind + ep_len]
        ep_returns = get_returns(ep_rewards, discount=discount)
        returns.extend(ep_returns)
        start_ind += ep_len
    returns = np.array(returns)

    return obs, next_obs, actions, action_log_probs, baselines, returns, rewards

def get_returns(
    rewards, discount=0.99,
):
    rollout_len = rewards.shape[0]
    returns = [[0] for _ in range(rollout_len)]

    for t in reversed(range(rollout_len)):
        if t == rollout_len-1:
            returns[t][0] = rewards[t][0]
        else:
            returns[t][0] = rewards[t][0] + discount*returns[t+1][0]

    return returns
