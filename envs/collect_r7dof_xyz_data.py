import numpy as np
from r7dof_xyz import R7DOFEnv
import pickle

n_trajs = 40
data = {}

for task in range(3):
    env = R7DOFEnv(task)
    data[task] = {
        'obs': [],
        'actions': [],
    }
    for traj in range(n_trajs):
        target = np.random.uniform(low=[-0.4, -0.4, -0.3], high=[0.4, 0.0, -0.3])
        obs = env.reset(target)
        for t in range(30):
            data[task]['obs'].append(obs)
            action = np.zeros(7, dtype=np.float32)
            action[0:3] = (target - env.get_body_com('end_eff')) / 5
            data[task]['actions'].append(action)
            obs, reward, done, info = env.step(action)
            #env.render()
            #import time; time.sleep(0.1)
        env.close()
    for (k, v) in data[task].items():
        data[task][k] = np.array(v)

pickle.dump(data, open('/home/ubuntu/LRMBMRL/data/r7dof/expert_trajs_xyz.pkl', 'wb'))
