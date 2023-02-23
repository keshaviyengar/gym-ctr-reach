import gym
import numpy as np

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


# Script to load a pretrained policy for 3 tube CTR and continue in hardware
if __name__ == '__main__':
    model_path = '/home/keshav/catkin_ws/src/ctr/ctr_policy_ros/example_model/rvim_hardware/her/' \
                 'CTR-Reach-v0_1/rl_model_3000000_steps.zip'
    env_kwargs = {
        'goal_tolerance_parameters': {
            'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001
        },
    }
    env, model = load_agent("CTR-Reach-Hardware-v0", env_kwargs, model_path)

    model.learn(total_timesteps=10, log_interval=1)
    model.save('her_ctr_reach_hardware')
    env.close()
