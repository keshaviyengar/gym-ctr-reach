import gym
import numpy as np
import ctr_reach_envs
from ctr_reach_envs.envs.model import Model

from stable_baselines.her.utils import HERGoalEnvWrapper

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 15

'''
The script below plots the selected ctr systems in the x-z plane to visualize the differences in tube parameters of the 
different systems.
'''

env_id = "CTR-Reach-v0"


def plot_ctr_systems(select_systems):
    """
    Given selected systems, render the tubes curvatures in x-z plae.
    :param select_systems: The selected systems to render.
    """
    env_kwargs = {'select_systems': select_systems}
    ctr_env = HERGoalEnvWrapper(gym.make(env_id, **env_kwargs)).env.env
    ctr_systems = ctr_env.ctr_system_parameters
    ctr_kine_model = Model(ctr_systems)

    fig = plt.figure()
    ax = plt.axes()

    # List of joints to visualize in workspace plot
    ext = ctr_env.trig_obj.joint_spaces[0].high[:3]
    joint_list = [np.concatenate((ext, np.array([ np.pi / 2,  np.pi / 2, np.pi / 2]))),
                  np.concatenate((ext, np.array([-np.pi / 2,  np.pi / 2, np.pi / 2]))),
                  np.concatenate((ext, np.array([ np.pi / 2, -np.pi / 2, np.pi / 2]))),
                  np.concatenate((ext, np.array([-np.pi / 2, -np.pi / 2, np.pi / 2]))),
                  ]
    #joint_list = [np.array([0, 0, 0,  np.pi / 2,  np.pi / 2, np.pi / 2]),
    #              np.array([0, 0, 0, -np.pi / 2,  np.pi / 2, np.pi / 2]),
    #              np.array([0, 0, 0,  np.pi / 2, -np.pi / 2, np.pi / 2]),
    #              np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, np.pi / 2])]
    # For each system, for each joint in the joint list, compute forward kinematics, get the backbone shape and plot
    for system in range(0, len(ctr_systems)):
        labelled = False
        for joint in joint_list:
            ee_pos = ctr_kine_model.forward_kinematics(joint, system)
            r1 = ctr_kine_model.r1
            r2 = ctr_kine_model.r2
            r3 = ctr_kine_model.r3
            if not labelled:
                labelled = True
                ax.plot(r1[:,0] * 1000, r1[:,2] * 1000, linewidth=4.0, c=plt.cm.tab20c(system*4), label='System ' + str(system))
            else:
                ax.plot(r1[:,0] * 1000, r1[:,2] * 1000, linewidth=4.0, c=plt.cm.tab20c(system*4))
            ax.plot(r2[:,0] * 1000, r2[:,2] * 1000, linewidth=6.0, c=plt.cm.tab20c(system*4 + 1))
            ax.plot(r3[:,0] * 1000, r3[:,2] * 1000, linewidth=8.0, c=plt.cm.tab20c(system*4 + 2))
            ax.set_xlabel('$x$ (mm)')
            ax.set_ylabel('$z$ (mm)')
            ax.set_xlim([-100, 250])
            ax.set_ylim([0, 400])
            ax.set_aspect('equal')
    ax.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    select_systems = [3]
    plot_ctr_systems(select_systems)

