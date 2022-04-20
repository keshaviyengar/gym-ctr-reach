from pynput.keyboard import Key, Listener
import gym
import time
import numpy as np

import ctr_generic_envs
from trajectory_plotter import load_agent, plot_trajectory


class KeyboardControl(object):
    def __init__(self, env):
        self.env = env

        self.key_listener = Listener(on_press=self.on_press_callback)
        self.key_listener.start()

        self.action = np.zeros_like(self.env.action_space.low)
        self.extension_actions = np.zeros(3)
        self.rotation_actions = np.zeros(3)

        self.extension_value = self.env.action_space.high[0] / 2
        self.rotation_value = self.env.action_space.high[-1] / 2
        self.exit = False

    def on_press_callback(self, key):
        # Tube 1 (inner most tube) is w s a d
        # Tube 2 (outer most tube) is t g f h
        # Tube 3 (outer most tube) is i k j l
        try:
            if key.char in ['w', 's', 'a', 'd']:
                if key.char == 'w':
                    self.extension_actions[0] = self.extension_value
                elif key.char == 's':
                    self.extension_actions[0] = -self.extension_value
                elif key.char == 'a':
                    self.rotation_actions[0] = self.rotation_value
                elif key.char == 'd':
                    self.rotation_actions[0] = -self.rotation_value
            if key.char in ['t', 'g', 'f', 'h']:
                if key.char == 't':
                    self.extension_actions[1] = self.extension_value
                elif key.char == 'g':
                    self.extension_actions[1] = -self.extension_value
                elif key.char == 'f':
                    self.rotation_actions[1] = self.rotation_value
                elif key.char == 'h':
                    self.rotation_actions[1] = -self.rotation_value
            if key.char in ['i', 'k', 'j', 'l']:
                if key.char == 'i':
                    self.extension_actions[2] = self.extension_value
                elif key.char == 'k':
                    self.extension_actions[2] = -self.extension_value
                elif key.char == 'j':
                    self.rotation_actions[2] = self.rotation_value
                elif key.char == 'l':
                    self.rotation_actions[2] = -self.rotation_value
        except AttributeError:
            if key == Key.esc:
                self.exit = True
                exit()
            else:
                self.extension_actions = np.zeros(3)
                self.rotation_actions = np.zeros(3)

    def run(self):
        obs = self.env.reset()
        while not self.exit:
            self.action[:3] = self.extension_actions
            self.action[3:] = self.rotation_actions
            # print('action: ', self.action)
            observation, reward, done, info = self.env.step(self.action)
            print(str(info['q_achieved'][3:]))
            self.extension_actions = np.zeros(3)
            self.rotation_actions = np.zeros(3)
            self.action = np.zeros_like(self.env.action_space.low)
            time.sleep(0.1)
        self.env.close()


if __name__ == '__main__':
    gen_model_path = "/her/CTR-Generic-Reach-v0_1/best_model.zip"

    project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/rotation_experiments/'
    names = ['constrain_rotation/tro_constrain_3', 'free_rotation/tro_free_3']
    #project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/generic_policy_experiments/'
    #names = ['two_tubes/tro_two_systems_2', 'three_tubes/tro_three_systems_0', 'four_tubes/tro_four_systems_0']
    system_idx = None
    exp = 1

    model_path = project_folder + names[exp] + gen_model_path
    # Env and model names and paths
    env_id = "CTR-Generic-Reach-v0"
    env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'constrain_alpha': False,
                  'num_systems': 1, 'select_systems': [3],
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001}
                  }
    env, _ = load_agent(env_id, env_kwargs, model_path)

    keyboard_agent = KeyboardControl(env)
    keyboard_agent.run()