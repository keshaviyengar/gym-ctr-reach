from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import numpy as np
import pandas as pp

from ctr_reach_envs.envs.CTR_Python.inverse_kinematics import JacobianIk
from ctr_reach_envs.src.plotting_utils import plot_path_only, compare_paths_only, plot_intermediate, plot_trajectory
from paths import line_traj, circle_traj, polygon_traj, helix_traj, velocity_based_line_traj, velocity_based_circle_traj
from ctr_reach_envs.envs.CTR_Python.Tube import Tube
from ctr_reach_envs.envs.CTR_Python.CTR_Model import CTR_Model
from ctr_reach_envs.envs.model import Model

from ctr_reach_envs.src.utils import load_agent, run_episode, trajectory_controller


def get_intial_starting_q(model_path, env_kwargs, env_id, x_0):
    # Use RL to finding starting position of first point
    env, model = load_agent(env_id, env_kwargs, model_path)
    ags, dgs, q, _, _, _ = run_episode(env, model, goal=x_0)
    return q


def path_following_RL(model_path, env_kwargs, env_id, path_array):
    x_0 = path_array[0,:]
    #q = get_intial_starting_q(model_path, env_kwargs, env_id, x_0)
    env, model = load_agent(env_id, env_kwargs, model_path)
    selected_systems = [3]

    # Run through trajectory controller and save goals and shape
    achieved_goals, desired_goals, r1, r2, r3 = trajectory_controller(model, env, path_array, 0, selected_systems)
    return achieved_goals, desired_goals


if __name__ == '__main__':
    # Get tube parameters from RL model directly
    exp_name = 'ral_constrain'
    model_path = '/home/keshav/ral_2023_results/results/ral-2023/' + exp_name + \
                 '/her/CTR-Reach-v0_1/rl_model_3000000_steps.zip'
    env_kwargs = {
        'resample_joints': False, 'evaluation': True,
        'constrain_alpha': False,
        'goal_tolerance_parameters': {
            'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001, 'measure': 'mm'
        }
    }
    env_id = 'CTR-Reach-v0'
    env, model = load_agent(env_id, env_kwargs, model_path)
    system_parameters = env.env.env.model.system_parameters
    home_offset = env.env.env.home_offset
    max_retraction = env.env.env.max_retraction

    # Setup model based Jacobian
    q = env.env.env.starting_joints
    CTR = Model(system_parameters)
    J = CTR.jac(q, 0)
    x_d = CTR.forward_kinematics(q, 0)

    # Get a path to follow
    path_type = 'line'
    if path_type == 'line':
        # x_points, y_points, z_points = line_traj(20, -0.1, 0.05, 0.20, 0.1, 0.05, 0.20)
        v = [0.0, 0.0005, 0.0005]
        #x_points, y_points, z_points = line_traj(10, -0.03, -0.04, 0.06, -0.03, -0.03, 0.08)
        x_points, y_points, z_points = velocity_based_line_traj(5, 10, v, x_d[0], x_d[1], x_d[2])
        #x_points, y_points, z_points = line_traj(10, 0.03, -0.03, 0.07, -0.03, -0.03, 0.07)
    if path_type == 'helix':
        x_points, y_points, z_points = helix_traj(100, 3, 0.03, 0.005, [0.06, 0.06, 0.18])
    if path_type == 'circle':
        x_points, y_points, z_points = velocity_based_circle_traj(50, 30, 0.10, 0.01, x_d[0], x_d[1], x_d[2])
        # x_points, y_points, z_points = circle_traj(40, 0.0, 0.10, 0.20, 0.05)

    # Concatenate into valid path for Jacobian solver
    path_array = np.vstack((x_points, y_points, z_points)).T

    achieved_goals, desired_goals = path_following_RL(model_path, env_kwargs, env_id, path_array)
    print('DeepRL errors: ')
    print(str(np.mean(np.linalg.norm((np.array(achieved_goals) - np.array(desired_goals)) * 1000, axis=1))))
    print(str(np.std(np.linalg.norm((np.array(achieved_goals) - np.array(desired_goals)) * 1000, axis=1))))
    plot_path_only(achieved_goals, desired_goals)

    # Control parameters
    K_p = 1.0
    damping_constant = 0.0
    jacobian_ik = JacobianIk(system_parameters, K_p, damping_constant, False, home_offset, max_retraction)
    # Path array is (n,3) matrix
    print("Following path with Jacobian.")
    # Get to starting position first
    env, model = load_agent(env_id, env_kwargs, model_path)
    _, _, q, _, _, _ = run_episode(env, model, goal=path_array[0,:])
    x_d_array, x_c_array, q_array = jacobian_ik.path_following(path_array, q)
    # Print error metrics
    print('Jacobian errors: ')
    print(str(np.mean(np.linalg.norm((x_c_array - x_d_array) * 1000, axis=1))))
    print(str(np.std(np.linalg.norm((x_c_array - x_d_array) * 1000, axis=1))))
    plot_path_only(x_c_array, x_d_array)

    compare_paths_only(achieved_goals, x_c_array, desired_goals)
