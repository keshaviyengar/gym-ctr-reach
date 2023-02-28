import gym
import ctr_reach_envs
import numpy as np

from stable_baselines import DDPG, HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper

from utils import load_agent, run_episode
from plotting_utils import plot_trajectory, animate_trajectory, generic_animate_trajectory

# Script takes an environment for CTRs and plots the followed trajectory with the CTR system
# 1. Load in agent
# 2. Run an episode
# 3. Track desired, achieved goals as well as robot shape
# 4. Create an animation of the robot

if __name__ == '__main__':
    gen_model_path = "/her/CTR-Reach-v0_1/rl_model_1800000_steps.zip"
    project_folder = '/home/keshav/ral_2023_results/new_extension/'
    name = 'ral_constrain'
    selected_systems = [0]

    animate = False
    plot_traj = True

    model_path = project_folder + name + gen_model_path
    output_path = project_folder + name

    noisy_env = False
    if noisy_env:
        noise_parameters = {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(1.0), 'extension_std': 0.001 * np.deg2rad(1.0), 'tracking_std': 0.0008
        }
    else:
        noise_parameters = {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'tracking_std': 0.0
        }

    # Env and model names and paths
    env_id = "CTR-Reach-v0"
    env_kwargs = {'evaluation': True, 'joint_representation': 'egocentric', 'resample_joints': True,
                  'constrain_alpha': False,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001,
                                                'measure': 'mm'},
                  'noise_parameters': noise_parameters,
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)
    achieved_goals, desired_goals, _, r1, r2, r3 = run_episode(env, model)
    #error = 0
    #while error < 2:
    #    achieved_goals, desired_goals, _, r1, r2, r3 = run_episode(env, model)
    #    error = np.linalg.norm(achieved_goals[-1] - desired_goals[-1]) * 1000
    #    print(error)
    if animate:
        ani = animate_trajectory(achieved_goals, desired_goals, r1, r2, r3, training_step=3e6, title=True)
        ani.save(output_path + '/ik.mp4', fps=5)
    if plot_traj:
        #plot_trajectory(achieved_goals, desired_goals, r1, r2, r3, save_path=output_path + '/ik.png')
        plot_trajectory(achieved_goals, desired_goals, r1, r2, r3)
