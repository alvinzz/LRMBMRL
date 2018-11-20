import numpy as np
import random as rd
from gym.envs.mujoco.mujoco_env import MujocoEnv
from PIL import Image
from gym import spaces
import cv2

class R7DOFEnv(MujocoEnv):
    def __init__(self, goal_num=0, frame_skip=5, xml_file=None, distance_metric_order=None, distractors=True, imsize=64, imsave_path=None, *args, **kwargs):
        self.goal_num = goal_num
        self.shuffle_order = [[0,1,2],[1,2,0],[2,0,1]][self.goal_num]

        self.imsize = imsize
        self.include_distractors=distractors
        assert distractors==True, "not supported"

        if xml_file is None:
            xml_file = '/home/ubuntu/LRMBMRL/envs/assets/reacher_7dof_2distr_%s%s%s.xml'%tuple(self.shuffle_order)

        self.frame_skip = frame_skip
        MujocoEnv.__init__(self, xml_file, self.frame_skip)
        # set properties
        self.obs_dim = self.get_current_image_obs()[1].size
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high)
        low = -np.inf * np.ones(self.obs_dim)
        high = np.inf * np.ones(self.obs_dim)
        self.observation_space = spaces.Box(low=low, high=high)

        self.im_num = 0
        self.window_name = 'dummy_r7dof_oracle'

    def get_current_image_obs(self):
        image = self.sim.render(self.imsize, self.imsize, camera_name='maincam')
        image = image[:, ::-1, [2,1,0]]
        image = image.astype(np.float32)
        state = self._get_obs()
        return image, state

    def step(self, action):
        distance = np.linalg.norm(self.get_body_com("tips_arm") - self.get_body_com("goal"))
        reward = - distance
        self.do_simulation(action, self.frame_skip)
        next_img, next_obs = self.get_current_image_obs()
        done = False
        return next_obs, reward, done, {'img': next_img}

    def sample_goals(self, num_goals):
        return np.zeros(num_goals)

    def reset(self, reset_args=None,reset_im_num=True, **kwargs):
        # print("debug,asked to reset with reset_args", reset_args)
        if reset_im_num:
            self.im_num = 0
        qpos = np.copy(self.init_qpos)
        qvel = np.copy(self.init_qvel) + 0.0*self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        while True:
            #self.goal = np.random.uniform(low=[-0.4, -0.4, -0.3], high=[0.4, 0.0, -0.3])
            self.goal = reset_args
            self.distract1 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3])
            self.distract2 = np.random.uniform(low=[-0.4,-0.4,-0.3],high=[0.4,0.0,-0.3])
            if np.linalg.norm(self.goal-self.distract1)>0.35 \
            and np.linalg.norm(self.goal-self.distract2)>0.35 \
            and np.linalg.norm(self.distract2-self.distract1)>0.35:
                break
        qpos[-14:-11] = self.distract1
        qpos[-21:-18] = self.distract2
        qpos[-7:-4] = self.goal
        qvel[-7:] = 0
        self.set_state(qpos, qvel)
        return self.get_current_image_obs()[1]

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.flat[:7],
            self.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            #self.get_body_com('goal'),
        ])

    def clip_goal_from_obs(self, paths):
        clip_dim = self.get_body_com('goal').shape[0]
        for path in paths:
            images = path['env_infos']['img']
            obs = path['observations'][:, :-clip_dim]
            #obs = path['observations']
            images = np.reshape(images, [-1, self.imsize*self.imsize*3])
            #path['observations'] = np.concatenate((obs, images), axis=-1)
            path['observations'] = np.concatenate((images, obs), axis=-1)
            del path['env_infos']
        return paths

    def render(self):
        pass

    def log_diagnostics(self, paths=[], prefix=''):
        pass
