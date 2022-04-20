import gym
import numpy as np
import ctr_reach_envs
from ctr_reach_envs.envs.model import Model

from stable_baselines.her.utils import HERGoalEnvWrapper

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15

# Given joint values and system, this script will plot the robot shape


if __name__ == '__main__':
    env_id = "CTR-Reach-v0"
    env_kwargs = {'select_systems': [0,1,2,3]}
    ctr_env =  HERGoalEnvWrapper(gym.make(env_id, **env_kwargs)).env.env
    ctr_systems = ctr_env.ctr_system_parameters
    ctr_kine_model = Model(ctr_systems)

    fig = plt.figure()
    ax = plt.axes()
    q_list = [np.array([0,0,0,np.pi/2,np.pi/2,np.pi/2]),
         np.array([0, 0, 0, -np.pi / 2, np.pi / 2, np.pi / 2]),
         np.array([0, 0, 0, np.pi / 2, -np.pi / 2, np.pi / 2]),
         np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, np.pi / 2])]
    for system_idx in range(0, len(ctr_systems)):
        labelled = False
        for q in q_list:
            ee_pos = ctr_kine_model.forward_kinematics(q, system_idx)
            r1 = ctr_kine_model.r1
            r2 = ctr_kine_model.r2
            r3 = ctr_kine_model.r3
            if not labelled:
                labelled = True
                ax.plot(r1[:,0] * 1000, r1[:,2] * 1000, linewidth=4.0, c=plt.cm.tab20c(system_idx*4), label='System ' + str(system_idx))
            else:
                ax.plot(r1[:,0] * 1000, r1[:,2] * 1000, linewidth=4.0, c=plt.cm.tab20c(system_idx*4))
            ax.plot(r2[:,0] * 1000, r2[:,2] * 1000, linewidth=6.0, c=plt.cm.tab20c(system_idx*4 + 1))
            ax.plot(r3[:,0] * 1000, r3[:,2] * 1000, linewidth=8.0, c=plt.cm.tab20c(system_idx*4 + 2))
            ax.set_xlabel('$x$ (mm)')
            ax.set_ylabel('$z$ (mm)')
            ax.set_xlim([-100, 250])
            ax.set_ylim([0, 400])
            ax.set_aspect('equal')
    ax.legend(loc='best')
    plt.grid()
    plt.show()
