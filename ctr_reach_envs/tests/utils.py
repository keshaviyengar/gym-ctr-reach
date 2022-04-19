import gym
import ctr_generic_envs
import numpy as np

from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

# Utility functions for loading models / agents and running episodes and trajectories

def decay_goal_tolerance(training_step):
    init_tol = 0.020
    final_tol = 0.001
    N_ts = 1.5e6
    r = 1 - np.power((final_tol / init_tol), 1 / N_ts)
    if training_step <= N_ts:
        return init_tol * np.power(1 - r, training_step)
    else:
        return 0.001

def load_agent(env_id, env_kwargs, model_path, seed=None):
    if seed is None:
        seed = np.random.randint(0, 10)
        set_global_seeds(seed)
    env = HERGoalEnvWrapper(gym.make(env_id, **env_kwargs))
    model = HER.load(model_path, env=env)
    return env, model

def run_episode(env, model, goal=None, system_idx=None):
    if goal is not None:
        if system_idx is not None:
            obs = env.reset(**{'goal': goal, 'system_idx': system_idx})
        else:
            obs = env.reset(**{'goal': goal})
    else:
        if system_idx is not None:
            obs = env.reset(**{'system_idx': system_idx})
        else:
            obs = env.reset()
    achieved_goals = list()
    desired_goals = list()
    r1 = list()
    r2 = list()
    r3 = list()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        obs_dict = env.convert_obs_to_dict(obs)
        achieved_goals.append(obs_dict['achieved_goal'])
        desired_goals.append(obs_dict['desired_goal'])
        r1.append(env.env.model.r1)
        r2.append(env.env.model.r2)
        r3.append(env.env.model.r3)
        # After each step, store achieved goal as well as rs
        if done or infos.get('is_success', False):
            print("Tip Error: " + str(infos.get('errors_pos')*1000))
            print("Achieved joint: " + str(np.rad2deg(infos.get('q_achieved'))))
            break
    return achieved_goals, desired_goals, r1, r2, r3

def trajectory_controller(model, env, x_traj, y_traj, z_traj, system_idx, select_systems):
    achieved_goals = list()
    desired_goals = list()
    r1 = list()
    r2 = list()
    r3 = list()
    # Get to first point in trajectory then start recording
    goal = np.array([x_traj[0], y_traj[0], z_traj[0]])
    if len(select_systems) > 1:
        obs = env.reset(**{'system_idx': np.where(system_idx == np.array(select_systems))[0][0],
                           'goal': goal})
    else:
        obs = env.reset(**{'goal': goal})
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        obs_dict = env.convert_obs_to_dict(obs)
        # After each step, store achieved goal as well as rs
        if done or infos.get('is_success', False):
            break

    achieved_goals.append(obs_dict['achieved_goal'])
    desired_goals.append(obs_dict['desired_goal'])
    r1.append(env.env.model.r1)
    r2.append(env.env.model.r2)
    r3.append(env.env.model.r3)

    for i in range(1, len(x_traj)):
        goal = np.array([x_traj[i], y_traj[i], z_traj[i]])
        if len(select_systems) > 1:
            obs = env.reset(**{'system_idx': np.where(system_idx == np.array(select_systems))[0][0],
                               'goal': goal})
        else:
            obs = env.reset(**{'goal': goal})
        # Set desired goal as x,y,z trajectory point in obs
        print(str(i) + ' out of ' + str(len(x_traj)))
        for _ in range(20):
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)
            obs_dict = env.convert_obs_to_dict(obs)
            achieved_goals.append(obs_dict['achieved_goal'])
            desired_goals.append(obs_dict['desired_goal'])
            r1.append(env.env.model.r1)
            r2.append(env.env.model.r2)
            r3.append(env.env.model.r3)
            # After each step, store achieved goal as well as rs
            if done or infos.get('is_success', False):
                break
    return achieved_goals, desired_goals, r1, r2, r3
