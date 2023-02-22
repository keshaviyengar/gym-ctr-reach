import gym
import ctr_reach_envs

import numpy as np
import pandas as pd
from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from utils import load_agent, trajectory_controller
from paths import line_traj, circle_traj, polygon_traj, helix_traj
from plotting_utils import animate_trajectory, plot_trajectory, plot_path_only, plot_intermediate

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


if __name__ == '__main__':
    gen_model_path = "/her/CTR-Reach-v0_1/rl_model_3000000_steps.zip"

    project_folder = '/home/keshav/ral_2023_results/results/ral-2023/'
    name = 'ral_constrain'
    selected_systems = [3]
    path_type = 'line'

    noisy_env = False
    plot_traj = True
    animate = False

    model_path = project_folder + name + gen_model_path
    output_path = project_folder + name

    if noisy_env:
        noise_parameters =  {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(1.0), 'extension_std': 0.001 * np.deg2rad(1.0), 'tracking_std': 0.0008
        }
    else:
        noise_parameters =  {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'tracking_std': 0.0
        }

    # Env and model names and paths
    env_id = "CTR-Reach-v0"
    env_kwargs = {'evaluation': True, 'joint_representation': 'egocentric', 'resample_joints': False, 'constrain_alpha': False,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001, 'measure': 'mm'},
                  'noise_parameters': noise_parameters,
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)

    # Get trajectory points
    if path_type == 'line':
        x_points, y_points, z_points = line_traj(10, -0.03, -0.04, 0.06, -0.03, -0.03, 0.08)
    if path_type == 'helix':
        x_points,y_points,z_points = helix_traj(100, 3, 0.03, 0.005, [0.06,0.06,0.18])
    if path_type == 'circle':
        # TODO: Net yet working
        x_points, y_points, z_points = circle_traj(20, 0.005, 0.0, 0.04, 0.0025)
    # Run through trajectory controller and save goals and shape
    path_array = np.vstack((x_points, y_points, z_points)).T
    achieved_goals, desired_goals, r1, r2, r3 = trajectory_controller(model, env, path_array, 0, selected_systems)

    errors = np.linalg.norm(np.array(achieved_goals) - np.array(desired_goals), axis=1)
    print('mean tracking error: ' + str(np.mean(errors) * 1000))
    print('std tracking error: ' + str(np.std(errors) * 1000))
    # Plot three trajectories intermediate
    if plot_traj:
        #plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, 0, output_path + '/' + path_type + '_1.png')
        #plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, int(len(achieved_goals) / 2), output_path + '/' + path_type + '_2.png')
        #plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, int(len(achieved_goals) - 1), output_path + '/' + path_type + '_3.png')
        ## Plot the path only
        #plot_path_only(achieved_goals, desired_goals, output_path + '/' + path_type + '_full.png')
        plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, int(len(achieved_goals) - 1))

    # Animate full trajectory
    if animate:
        ani = animate_trajectory(achieved_goals, desired_goals, r1, r2, r3, title=False)
        if noisy_env:
            ani.save(output_path + '/noisy_traj_' + path_type + '.mp4', fps=10)
        else:
            ani.save(output_path + '/traj_' + path_type + '.mp4', fps=10)
