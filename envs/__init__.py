import logging

from gym.envs import register
import numpy as np
import pickle

LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(id='ObjPusher-v0', entry_point='LRMBMRL.envs.pusher_env:PusherEnv', kwargs={'sparse_reward': False})
    register(id='TwoDMaze-v0', entry_point='LRMBMRL.envs.twod_maze:TwoDMaze')
    register(id='PointMazeRight-v0', entry_point='LRMBMRL.envs.point_maze_env:PointMazeEnv',
            kwargs={'sparse_reward': False, 'direction': 1})
    register(id='PointMazeLeft-v0', entry_point='LRMBMRL.envs.point_maze_env:PointMazeEnv',
            kwargs={'sparse_reward': False, 'direction': 0})

    # pointmass
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR01Goals.pkl', 'rb'))[:15]
    register(id='PointMassR01-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR01Goals.pkl', 'rb'))[-5:]
    register(id='PointMassR01_test-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR02Goals.pkl', 'rb'))[:15]
    register(id='PointMassR02-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR02Goals.pkl', 'rb'))[-5:]
    register(id='PointMassR02_test-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR03Goals.pkl', 'rb'))[:15]
    register(id='PointMassR03-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR03Goals.pkl', 'rb'))[-5:]
    register(id='PointMassR03_test-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR12Goals.pkl', 'rb'))[:15]
    register(id='PointMassR12-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR23Goals.pkl', 'rb'))[:15]
    register(id='PointMassR23-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})
    tasks = pickle.load(open('envs/pointMassRadGoals/pointMassR34Goals.pkl', 'rb'))[:15]
    register(id='PointMassR34-v0', entry_point='LRMBMRL.envs.pointmass:PointMass',
            kwargs={'tasks': tasks})

    # A modified ant which flips over less and learns faster via TRPO
    register(id='CustomAnt-v0', entry_point='LRMBMRL.envs.ant_env:CustomAntEnv',
            kwargs={'gear': 30, 'disabled': False})
    register(id='DisabledAnt-v0', entry_point='LRMBMRL.envs.ant_env:CustomAntEnv',
            kwargs={'gear': 30, 'disabled': True})

    register(id='VisualPointMaze-v0', entry_point='LRMBMRL.envs.visual_pointmass:VisualPointMazeEnv',
            kwargs={'sparse_reward': False, 'direction': 1})
