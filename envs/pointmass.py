import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np

from envs.dynamic_mjc.mjc_models import pointmass

class PointMass(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, target=np.array([1., 0]), episode_length=20):
        utils.EzPickle.__init__(self)

        self.max_episode_length = episode_length
        self.target = target
        self.episode_length = -1

        model = pointmass(target)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, frame_skip=5)

        self.reset_model()

    def step(self, a):
        ob = self._get_obs()

        self.do_simulation(a, self.frame_skip)

        vec_dist = self.get_body_com("particle") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec_dist)**2
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + 0.001 * reward_ctrl

        self.episode_length += 1
        done = (self.episode_length >= self.max_episode_length)

        return ob, reward, done, {}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.1, high=0.1)
        self.episode_length = -1
        self.set_state(qpos, qvel)
        self.step([0., 0])
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("particle"),
            self.get_body_com("target"),
        ])

    def plot_trajs(self, *args, **kwargs):
        pass

    def log_diagnostics(self, paths):
        rew_dist = np.array([traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array([traj['env_infos']['reward_ctrl'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', -np.mean(rew_dist.mean()))
        logger.record_tabular('AvgControlCost', -np.mean(rew_ctrl.mean()))
        logger.record_tabular('AvgMinToGoalDist', np.mean(np.min(-rew_dist, axis=1)))
