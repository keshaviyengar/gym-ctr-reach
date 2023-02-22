import gym
import numpy as np
import time
import csv
import ctr_reach_envs

from stable_baselines import HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


def load_agent(env_id, env_kwargs, model_path, seed=None):
    if seed is None:
        seed = np.random.randint(0, 10)
        set_global_seeds(seed)
    env = HERGoalEnvWrapper(gym.make(env_id, **env_kwargs))
    model = HER.load(model_path, env=env)
    return env, model


def data_line(episode, step, joints, achieved_goal, desired_goal, error):
    return list(np.concatenate((np.array([episode]), np.array([step]), joints, achieved_goal, desired_goal, error)))


def drl_path_following(env, model, path_array, output_path):
    episode = 0

    args = {'goal': path_array[0, :]}
    obs = env.reset(**args)
    # Move to starting position
    for _ in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)
        obs_dict = env.convert_obs_to_dict(obs)
        # After each step, store achieved goal as well as rs
        print(infos)
        if done or infos.get('is_success', False):
            break

    for i in range(1, path_array.shape[0]):
        goal = np.array([path_x[i], path_y[i], path_z[i]])
        obs = env.reset(**{'goal': goal})
        # Set desired goal as x,y,z trajectory point in obs
        print(str(i) + ' out of ' + str(len(path_x)))
        with open(output_path, 'w+') as f:
            writer = csv.writer(f)
            for step in range(5):
                action, _ = model.predict(obs, deterministic=True)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, done, infos = env.step(action)
                obs_dict = env.convert_obs_to_dict(obs)
                writer.writerow(data_line(i, step, info['joints'], info['achieved_goal'],
                                          info['desired_goal'], np.array([info['error']])))
            # After each step, store achieved goal as well as rs
            print(infos.get('error'))
            if done or infos.get('is_success', False):
                break


def drl_ik_solver(env, model, episode, desired_goal=None, initial_joints=None, output_path=None):
    # if desired goal is None, sample randomly
    args = {}
    if initial_joints is not None:
        args['joints'] = initial_joints
    if desired_goal is not None:
        args['goal'] = desired_goal

    obs = env.reset(**args)
    # Move to starting position
    time.sleep(0.5)
    with open(output_path, 'w+') as f:
        writer = csv.writer(f)
        for step in range(20):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print('error: ' + str(info['error'] * 1000))
            print("action:  " + str(action))
            writer.writerow(data_line(episode, step, info['joints'], info['achieved_goal'],
                                      info['desired_goal'], np.array([info['error']])))
            time.sleep(1.0)
            if done:
                break
        print('ep: ' + str(episode))
        print('error: ' + str(info['error'] * 1000))
        return info['joints']


def jacobian_path_following(env, model, K_p, damping_constant, damping, desired_path, writer):
    system_parameters = env.env.env.model.system_parameters
    # Control parameters
    jacobian_ik = JacobianIk(system_parameters, K_p, damping_constant, False, home_offset, max_retraction)
    # Path array is (n,3) matrix
    # Get to starting position first
    env, model = load_agent(env_id, env_kwargs, model_path)
    _, _, q, _, _, _ = run_episode(env, model, goal=path_array[0, :])
    x_d_array, x_c_array, q_array = jacobian_ik.path_following(path_array, q)
    # Send desired joint values
    args = {}
    if initial_joints is not None:
        args['joints'] = q

    obs = env.reset(**args)
    # Move to starting position
    time.sleep(0.5)
    with open(output_path, 'w+') as f:
        writer = csv.writer(f)
        for step in range(q_array.shape[0]):
            action = q_array[step, :]
            obs, reward, done, info = env.step(action)
            writer.writerow(data_line(episode, step, info['joints'], info['achieved_goal'],
                                      info['desired_goal'], np.array([info['error']])))


if __name__ == '__main__':
    method = 'drl'
    path = None
    # path = 'line'
    if path is not None:
        solving = 'path_following'
    else:
        solving = 'ik'
    exp_name = 'ral_full_noise'

    home_offset = np.array([427.82e-3, 119.69e-3, 50.75e-3])
    max_retraction = np.array([97.0e-3, 50.0e-3, 22.0e-3])
    model_path = '/home/keshav/ral_2023_results/results/ral-2023/' + exp_name + '/her/CTR-Reach-v0_1/rl_model_3000000_steps.zip'
    # env_kwargs = {
    #    'resample_joints': False, 'evaluation': True,
    #    'constrain_alpha': False,
    #    'goal_tolerance_parameters': {
    #        'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
    #        'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001, 'measure': 'mm'
    #    }
    # }
    env_kwargs = {'resample_joints': False, 'initial_joints': np.concatenate((-home_offset, np.zeros(3))),
                  'goal_tolerance_parameters': {
                      'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
                      'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001, 'measure': 'mm'}
                  }
    env, model = load_agent("CTR-Reach-Hardware-v0", env_kwargs, model_path)
    header = ['episode', 'step', 'beta_0', 'beta_1', 'beta_2', 'alpha_1', 'alpha_2', 'alpha_3', 'ag_x', 'ag_y', 'ag_z',
              'dg_x', 'dg_y', 'dg_z', 'error']
    if path is not None:
        output_path = '/home/keshav/ral_2023_results/results/' + solving + '/' + method + '_' + path + '.csv'
    else:
        output_path = '/home/keshav/ral_2023_results/results/' + solving + '/' + exp_name + '.csv'
    print('output_path: ' + str(output_path))
    with open(output_path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(header)
    if path is not None:
        if path == 'line':
            v = [0.0, 0.0005, 0.0005]
            # x_points, y_points, z_points = line_traj(10, -0.03, -0.04, 0.06, -0.03, -0.03, 0.08)
            x_points, y_points, z_points = velocity_based_line_traj(5, 10, v, x_d[0], x_d[1], x_d[2])

        # Concatenate into valid path for Jacobian solver
        path_array = np.vstack((x_points, y_points, z_points)).T
        if method == 'drl':
            drl_path_following(env, model, path_array, output_path)
        elif method == 'jacobian':
            jacobian_path_following(env, model, K_p, damping_constant, False, path_array, output_path)
    else:
        # Number of trials
        episodes = 1
        for ep in range(episodes):
            desired_goal = None
            if method == 'drl':
                drl_ik_solver(env, model, episode=ep, output_path=output_path)
            elif method == 'jacobian':
                jacobian_ik_solver(env, model, K_p, damping_constant, False, desired_goal, output_path)
