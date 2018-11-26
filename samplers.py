import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
import copy

class MetaParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.
    Args:
        env_fns: meta-environments
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                             the respective environment is reset
    """

    def __init__(self, env_fns, envs_per_task, max_path_length):
        self.envs_per_task = envs_per_task
        self.env_fns = []
        for env_fn in env_fns:
            for _ in range(envs_per_task):
                self.env_fns.append(env_fn)

        self.meta_batch_size = len(env_fns)
        self.n_envs = self.meta_batch_size * envs_per_task
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.n_envs)])
        seeds = np.random.choice(range(10**6), size=self.n_envs, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, remote, env_fn, 1, max_path_length, seed))
            for (work_remote, remote, env_fn, seed) in zip(self.work_remotes, self.remotes, self.env_fns, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, mb_task_inds=None):
        """
        Executes actions on each env
        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task
        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of envs)
        """
        if mb_task_inds is None:
            assert len(actions) == self.n_envs
        else:
            assert len(actions) == len(mb_task_inds) * self.envs_per_task

        # step remote environments
        if mb_task_inds is not None:
            mb_remotes = []
            for task_ind in mb_task_inds:
                mb_remotes.extend(self.remotes[self.envs_per_task*task_ind:self.envs_per_task*(task_ind+1)])
        else:
            mb_remotes = self.remotes
        for remote, action in zip(mb_remotes, actions):
            remote.send(('step', [action]))

        results = [remote.recv() for remote in mb_remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return obs, rewards, dones, env_infos

    def reset(self, reset_args=None, mb_task_inds=None):
        """
        Resets the environments of each worker
        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        if mb_task_inds is not None:
            mb_remotes = []
            for task_ind in mb_task_inds:
                mb_remotes.extend(self.remotes[self.envs_per_task*task_ind:self.envs_per_task*(task_ind+1)])
        else:
            mb_remotes = self.remotes
        for remote, reset_arg in zip(mb_remotes, reset_args):
            remote.send(('reset', reset_arg))
        return sum([remote.recv() for remote in mb_remotes], [])

    @property
    def num_envs(self):
        """
        Number of environments
        Returns:
            (int): number of environments
        """
        return self.n_envs


def worker(remote, parent_remote, env_fn, n_envs, max_path_length, seed):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.
    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [env_fn() for _ in range(n_envs)]
    np.random.seed(seed)

    ts = np.zeros(n_envs, dtype='int')

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    infos[i]['next_obs'] = obs[i].copy()
                    obs[i] = envs[i].reset()
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))

        # reset all the environments of the worker
        elif cmd == 'reset':
            if data is None:
                obs = [env.reset() for env in envs]
            else:
                obs = [env.reset(data) for env in envs]
            ts[:] = 0
            remote.send(obs)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError

if __name__ == '__main__':
    from envs.r7dof import R7DOFEnv
    env_fns = [lambda: R7DOFEnv(i//20) for i in range(60)]
    envs_per_task = 1
    max_path_length = 30
    sampler = MetaParallelEnvExecutor(env_fns, envs_per_task, max_path_length)
    targets = np.random.uniform(low=[-0.4,-0.4,-0.3], high=[0.4,0,-0.3], size=(60, 3))
    import time
    start = time.time()
    obs = sampler.reset(targets)
    for t in range(30):
        sampled_actions = np.random.uniform(low=-np.ones(7), high=np.ones(7), size=(60, 7))
        obs, rewards, dones, infos = sampler.step(sampled_actions)
    end = time.time()
    print(end-start)
