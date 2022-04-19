import gym
import ctr_generic_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from trajectory_plotter import load_agent, plot_trajectory
from trajectory_generator import line_traj, circle_traj, polygon_traj, helix_traj

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from ctr_generic_envs.envs.CTR_Python import CTR_Model


def dls_ik_position_only(pos, q0, tube_systems, lam=0.25, num=500):
    q_0 = np.array([0, 0, 0, 0, 0, 0])
    # initial twist (for ivp solver)
    uz_0 = np.array([0.0, 0.0, 0.0])
    u1_xy_0 = np.array([[0.0], [0.0]])
    # force on robot tip along x, y, and z direction
    f = np.array([0, 0, 0]).reshape(3, 1)

    # Plotting
    # Setup plotting
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1,1,1])
    # keep track of ags for plotting
    ags = list()

    Pd = pos.ravel()
    q = q0.copy()
    # run loop trying to get to target:
    e = np.ones(3)
    for _ in range(num):
        CTR = CTR_Model(tube_systems[0], tube_systems[1], tube_systems[2], f, q, q_0, 0.01, 1)
        J = CTR.jac(np.concatenate((u1_xy_0, uz_0), axis=None))  # estimate jacobian matrix
        # Compute P, new position of robot
        P = CTR.r[-1, :]
        e = Pd - P.ravel()
        dq = np.array(np.matmul(J.T, np.linalg.inv(np.matmul(J, J.T) + lam * np.eye(J.shape[0]))))
        #dq = J.T
        q += np.matmul(dq, e)

        if np.linalg.norm(e) < 1e-3:
            break
        print("error: " + str(np.linalg.norm(e)))

        # Save to list of achieved goals
        ags.append(P)
        # Plot state of robot and goals
        ax.plot3D(CTR.r[:, 0], CTR.r[:, 1], CTR.r[:, 2], linewidth=2.0)
        ag = np.array(ags)
        ax.plot3D(ag[:, 0], ag[:, 1], ag[:, 2], marker='.', linestyle=':')
        ax.plot3D(pos[0], pos[1], pos[2], marker='.', linestyle=':')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-0.1, 0.1])
        ax.set_ylim([-0.1, 0.1])
        ax.set_zlim([0, 0.2])
        plt.draw()
        plt.pause(0.01)
        ax.cla()
        ##############################################################
        # should add a break condition and corresponding tolerance!! #
        ##############################################################
    return q, np.linalg.norm(e)


if __name__ == '__main__':
    # Load CTR gym Environment.
    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/rotation_experiments/'
    gen_model_path = "/her/CTR-Generic-Reach-v0_1/rl_model_2000000_steps.zip"
    names = ['constrain_rotation/tro_constrain_0', 'free_rotation/tro_free_0']

    model_path = project_folder + names[0] + gen_model_path
    output_path = project_folder + names[0] + "/evaluations.csv"
    num_episodes = 500

    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'num_systems': 1,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001}
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)

    # Use the systems from the gym enviornment
    tube_systems = env.env.env.model.systems[0]

    q_0 = np.array([-0.003,  0.002, -0.001, 0.0, 0.0, 0.0])
    # initial twist (for ivp solver)
    uz_0 = np.array([0.0, 0.0, 0.0])
    u1_xy_0 = np.array([[0.0], [0.0]])
    # force on robot tip along x, y, and z direction
    f = np.array([0, 0, 0]).reshape(3, 1)
    CTR = CTR_Model(tube_systems[0], tube_systems[1], tube_systems[2], f, q_0, q_0, 0.01, 1)
    CTR.ode_solver(np.concatenate((u1_xy_0, uz_0), axis=None))
    ag = CTR.r[-1, :]

    dg = ag + np.random.rand(3) / 200
    print(np.linalg.norm(dg - ag))

    # Solve IK with DLS
    q, error = dls_ik_position_only(dg, q_0, tube_systems)
