import gym
import ctr_reach_envs
import numpy as np
from ctr_reach_envs.envs import CtrReachEnv

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

def run_episode(env, model, goal=None, system_idx=None, max_steps=None):
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
    qs = list()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        obs_dict = env.convert_obs_to_dict(obs)
        achieved_goals.append(obs_dict['achieved_goal'])
        desired_goals.append(obs_dict['desired_goal'])
        if isinstance(env.env.env, CtrReachEnv):
            r1.append(env.env.env.model.r1)
            r2.append(env.env.env.model.r2)
            r3.append(env.env.env.model.r3)
        else:
            r1.append(env.env.env.env.model.r1)
            r2.append(env.env.env.env.model.r2)
            r3.append(env.env.env.env.model.r3)
        qs.append(infos['q_achieved'])
        # After each step, store achieved goal as well as rs
        if done or infos.get('is_success', False):
            print("Tip Error: " + str(infos.get('error')*1000))
            print("Achieved joint: " + str(infos.get('q_achieved')))
            break
    if infos.get('errors_pos') > 0.005:
        print("Could not get close to starting position...")
    return achieved_goals, desired_goals, qs, r1, r2, r3

def trajectory_controller(model, env, path_array, system_idx, select_systems):
    achieved_goals = list()
    desired_goals = list()
    r1 = list()
    r2 = list()
    r3 = list()
    qs = list()
    # Get to first point in trajectory then start recording
    goal = path_array[0, :]
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
    if infos['errors_pos'] > 0.005:
        print("Could not get close to starting position... error: " + str(infos.get('errors_pos')))
        return
    achieved_goals.append(obs_dict['achieved_goal'])
    desired_goals.append(obs_dict['desired_goal'])
    if isinstance(env.env.env, CtrReachEnv):
        r1.append(env.env.env.model.r1)
        r2.append(env.env.env.model.r2)
        r3.append(env.env.env.model.r3)
    else:
        r1.append(env.env.env.env.model.r1)
        r2.append(env.env.env.env.model.r2)
        r3.append(env.env.env.env.model.r3)
    qs.append(infos['q_achieved'])
    eps = list()
    steps = list()

    for i in range(1, path_array.shape[0]):
        eps.append(i)
        goal = path_array[i, :]
        if len(select_systems) > 1:
            obs = env.reset(**{'system_idx': np.where(system_idx == np.array(select_systems))[0][0],
                               'goal': goal})
        else:
            obs = env.reset(**{'goal': goal})
        # Set desired goal as x,y,z trajectory point in obs
        print(str(i) + ' out of ' + str(path_array.shape[0]))
        for step in range(2):
            steps.append(step)
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, infos = env.step(action)
            obs_dict = env.convert_obs_to_dict(obs)
            achieved_goals.append(obs_dict['achieved_goal'])
            desired_goals.append(obs_dict['desired_goal'])
            if isinstance(env.env.env, CtrReachEnv):
                r1.append(env.env.env.model.r1)
                r2.append(env.env.env.model.r2)
                r3.append(env.env.env.model.r3)
            else:
                r1.append(env.env.env.env.model.r1)
                r2.append(env.env.env.env.model.r2)
                r3.append(env.env.env.env.model.r3)
            qs.append(infos['q_achieved'])
            # After each step, store achieved goal as well as rs
            if done or infos.get('is_success', False):
                break
        print('error: ' + str(infos['errors_pos']))
    return achieved_goals, desired_goals, qs, r1, r2, r3, eps, steps
