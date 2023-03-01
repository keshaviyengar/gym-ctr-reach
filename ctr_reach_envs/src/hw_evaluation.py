import gym
import numpy as np
import time
import csv
import ctr_reach_envs
from paths import *
from ctr_reach_envs.envs.CTR_Python.inverse_kinematics import JacobianIk
from ctr_reach_envs.src.utils import load_agent, run_episode, trajectory_controller

from datetime import datetime
import os

from stable_baselines import HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


def data_line(episode, step, joints, achieved_goal, desired_goal, error):
    return list(np.concatenate((np.array([episode]), np.array([step]), joints, achieved_goal, desired_goal, error)))


def drl_path_following(env, model, path_array, output_path):
    achieved_goals, desired_goals, qs, r1, r2, r3, eps, steps = trajectory_controller(model, env, path_array, 0, [3])
    with open(output_path, 'a') as f:
        writer = csv.writer(f)
        for ag, dg, joint, ep, step in zip(achieved_goals, desired_goals, qs, eps, steps):
            error = np.linalg.norm(ag - dg)
            writer.writerow(data_line(ep, step, joint, ag, dg, np.array([error])))
    return achieved_goals, desired_goals, r1, r2, r3, qs, eps, steps


def drl_ik_solver(env, model, episode, desired_goal=None, initial_joints=None, output_path=None):
    # if desired goal is None, sample randomly
    args = {}
    if initial_joints is not None:
        args['joints'] = initial_joints
    if desired_goal is not None:
        args['goal'] = desired_goal

    # Move to starting position
    time.sleep(0.5)
    achieved_goals, desired_goals, qs, r1, r2, r3 = run_episode(env, model, desired_goal)
    with open(output_path, 'a') as f:
        writer = csv.writer(f)
        step = 0
        for ag, dg, joint in zip(achieved_goals, desired_goals, qs):
            error = np.linalg.norm(ag - dg)
            step += 1
            writer.writerow(data_line(episode, step, joint, ag, dg, np.array([error])))
    return achieved_goals, desired_goals, qs, r1, r2, r3


def jacobian_path_following(env, model, K_p, damping_constant, damping, path_array, output_path):
    if isinstance(env.env.env, ctr_reach_envs.envs.CtrReachHardwareEnv):
        system_parameters = env.env.env.env.model.system_parameters
        starting_q = env.env.env.env.trig_obj.joints
    else:
        system_parameters = env.env.env.model.system_parameters
        starting_q = env.env.env.trig_obj.joints
    # Control parameters
    jacobian_ik = JacobianIk(system_parameters, K_p, damping_constant, damping, home_offset, max_retraction)
    # Path array is (n,3) matrix
    # Get to starting position first
    _, _, qs, _, _, _ = run_episode(env, model, goal=path_array[0, :])
    x_d_array, x_c_array, q_array = jacobian_ik.path_following(path_array, qs[-1])

    obs = env.reset()
    # Move to starting position
    time.sleep(0.5)
    episode = 0
    with open(output_path, 'w+') as f:
        writer = csv.writer(f)
        for step in range(q_array.shape[0]):
            action = q_array[step, :]
            obs, reward, done, info = env.step(action)
            writer.writerow(data_line(episode, step, info['joints'], info['achieved_goal'],
                                      info['desired_goal'], np.array([info['error']])))
    return x_d_array, x_c_array, q_array


def jacobian_ik_solver(env, model, K_p, damping_constant, damping, desired_goal, output_path):
    if isinstance(env.env.env, ctr_reach_envs.envs.CtrReachHardwareEnv):
        system_parameters = env.env.env.env.model.system_parameters
        starting_q = env.env.env.env.trig_obj.joints
    else:
        system_parameters = env.env.env.model.system_parameters
        starting_q = env.env.env.trig_obj.joints

    obs = env.reset()
    obs_dict = env.convert_obs_to_dict(obs)
    if desired_goal is None:
        desired_goal = obs_dict['desired_goal']

    # Control parameters
    jacobian_ik = JacobianIk(system_parameters, K_p, damping_constant, damping, home_offset, max_retraction)
    x_d_array, x_c_array, q_array = jacobian_ik.ik_solver(desired_goal, starting_q)

    time.sleep(0.5)
    episode = 0
    with open(output_path, 'w+') as f:
        writer = csv.writer(f)
        for step in range(q_array.shape[0]):
            action = q_array[step, :]
            obs, reward, done, info = env.step(action)
            writer.writerow(data_line(episode, step, info['joints'], info['achieved_goal'],
                                      info['desired_goal'], np.array([info['error']])))
    return x_d_array, x_c_array, q_array


if __name__ == '__main__':
    method = 'drl'
    path = None
    #path = 'line'
    if path is not None:
        solving = 'path_following'
    else:
        solving = 'ik'

    exp_names = ['ral_full_noise_10', 'ral_full_noise_20']

    for exp_name in exp_names:
        home_offset = np.array([427.82e-3, 119.69e-3, 50.75e-3])
        max_retraction = np.array([97.0e-3, 50.0e-3, 22.0e-3])
        model_path = '/home/keshav/ral_2023_results/new_extension/' + exp_name + '/her/CTR-Reach-v0_1/rl_model_2000000_steps.zip'
        env_kwargs = {'resample_joints': False, 'initial_joints': np.concatenate((-home_offset, np.zeros(3))),
                      'goal_tolerance_parameters': {
                          'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
                          'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001, 'measure': 'mm'},
                      'evaluation': True, 'max_steps_per_episode': 10
                      }
        env, model = load_agent("CTR-Reach-Hardware-v0", env_kwargs, model_path)
        header = ['episode', 'step', 'beta_0', 'beta_1', 'beta_2', 'alpha_1', 'alpha_2', 'alpha_3', 'ag_x', 'ag_y', 'ag_z',
                  'dg_x', 'dg_y', 'dg_z', 'error']

        damping_constant = 0
        K_p = 2.0
        date_time = datetime.now().strftime("%m_%d_%H_%M_%S")
        if path is not None:
            output_dir = '/home/keshav/ral_2023_results/' + solving + '/' + date_time + '/'
            output_path = output_dir + method + '_' + path + '.csv'
        else:
            output_dir = '/home/keshav/ral_2023_results/' + solving + '/' + date_time + '/'
            output_path = output_dir + exp_name + '.csv'
        print('output_path: ' + str(output_path))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        if path is not None:
            if path == 'line':
                v = [0.0, 0.0005, 0.0005]
                x_points, y_points, z_points = line_traj(10, -0.03, -0.04, 0.06, -0.03, -0.03, 0.08)
                # Concatenate into valid path for Jacobian solver
                path_array = np.vstack((x_points, y_points, z_points)).T
                if method == 'drl':
                    drl_path_following(env, model, path_array, output_path)
                elif method == 'jacobian':
                    jacobian_path_following(env, model, K_p, damping_constant, False, path_array, output_path)
        else:
            # Number of trials
            episodes = 10
            for ep in range(episodes):
                desired_goal = None
                if method == 'drl':
                    drl_ik_solver(env, model, episode=ep, output_path=output_path)
                elif method == 'jacobian':
                    jacobian_ik_solver(env, model, K_p, damping_constant, False, desired_goal, output_path)
