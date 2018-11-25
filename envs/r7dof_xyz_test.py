from r7dof_xyz import R7DOFEnv as Env
import time
import numpy as np

e = Env(0)
obs = e.reset(np.array([0,0,-0.3]))
for t in range(30):
    #action = e.action_space.sample()
    #print(action)
    action = np.zeros(7)
    action[0:3] = [-0.1, 0.1, 0]
    e.step(action)
    e.render()
e.close()
