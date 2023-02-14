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


def path_following(env, model, writer):
    # Get path
    from paths import line_traj
    path_x, path_y, path_z = line_traj(10, 0.03, -0.03, 0.07, -0.03, -0.03, 0.07)
    episode = 0

    args = {'goal': np.array([path_x[0], path_y[0], path_z[0]])}
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

    for i in range(1, len(path_x)):
        goal = np.array([path_x[i], path_y[i], path_z[i]])
        obs = env.reset(**{'goal': goal})
        # Set desired goal as x,y,z trajectory point in obs
        print(str(i) + ' out of ' + str(len(path_x)))
        for _ in range(5):
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)
            obs_dict = env.convert_obs_to_dict(obs)
            # After each step, store achieved goal as well as rs
            print(infos.get('error'))
            if done or infos.get('is_success', False):
                break


def drl_ik_solver(env, model, episode, desired_goal=None, initial_joints=None, writer=None):
    args = {}
    if initial_joints is not None:
        args['joints'] = initial_joints
    if desired_goal is not None:
        args['goal'] = desired_goal

    obs = env.reset(**args)
    # Move to starting position
    time.sleep(0.5)
    for step in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if writer is not None:
            writer.writerow(data_line(episode, step, info['joints'], info['achieved_goal'],
                                      info['desired_goal'], np.array([info['error']])))
        time.sleep(1.0)
        if done:
            print("Reached goal!")
            print('ep: ' + str(episode))
            print('error: ' + str(info['error'] * 1000))
            return info['joints']


if __name__ == '__main__':
    home_offset = np.array([427.82e-3, 119.69e-3, 50.75e-3])
    max_retraction = np.array([97.0e-3, 50.0e-3, 22.0e-3])
    model_path = '/home/keshav/catkin_ws/src/ctr/ctr_policy_ros/example_model/tro_constrain_3/her/' \
                 'CTR-Reach-v0_1/rl_model_3000000_steps.zip'
    env_kwargs = {'resample_joints': False, 'initial_joints': np.concatenate((-home_offset, np.zeros(3))),
                  'goal_tolerance_parameters': {
                      'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
                      'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001
                  }
                  }
    env, model = load_agent("CTR-Reach-Hardware-v0", env_kwargs, model_path)
    header = ['episode', 'step', 'beta_0', 'beta_1', 'beta_2', 'alpha_1', 'alpha_2', 'alpha_3', 'ag_x', 'ag_y', 'ag_z',
              'dg_x', 'dg_y', 'dg_z', 'error']
    episodes = 10
    with open('hw_evaluation.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        #path_following(env, model, writer)
        for ep in range(episodes):
            drl_ik_solver(env, model, ep, writer=writer)
