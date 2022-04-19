# Iterate through models
# Complete a path following task
# Save the frames, include timestep number
# Go to next model and repeat

import numpy as np
from moviepy.editor import *
import matplotlib.pyplot as plt

from plotting_utils import animate_trajectory, generic_animate_trajectory
from utils import load_agent, run_episode, decay_goal_tolerance


create_animations = True
combine_mp4 = True

if __name__ == '__main__':
    file_names = list()
    for i in range(200000, 3000000+1, 200000):
        # Load in model
        gen_model_path = "/her/CTR-Generic-Reach-v0_1/rl_model_" + str(i) + "_steps.zip"
        project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/rotation_experiments/'
        #project_folder = '/home/keshav/ctm2-stable-baselines/saved_results/tro_2021/tro_results/generic_policy_experiments/'
        name = 'free_rotation/tro_free_3'
        #name = 'four_systems/tro_four_systems_0'
        selected_systems = [3]
        model_path = project_folder + name + gen_model_path
        output_path = project_folder + name

        env_id = "CTR-Generic-Reach-v0"
        env_kwargs = {'evaluation': True, 'relative_q': True, 'resample_joints': True, 'constrain_alpha': False,
                      'num_systems': len(selected_systems), 'select_systems': selected_systems,
                      'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                    'N_ts': 1.5e6, 'function': 'constant', 'set_tol': decay_goal_tolerance(i)},
                      }
        try:
            env, model = load_agent(env_id, env_kwargs, model_path)
            file_name = '/home/keshav/tmp_' + str(i) + '.mp4'
        except:
            print(model_path)
            print("File not found or environment and model mismatch...")
            continue
        file_names.append('/home/keshav/tmp_' + str(i) + '.mp4')
        if create_animations:
            if len(selected_systems) > 1:
                system = np.random.randint(0, len(selected_systems))
                # Run a episode reaching a goal
                achieved_goals, desired_goals, r1, r2, r3 = run_episode(env, model, system_idx=system)
                print('Tip Error (): ' + str(np.linalg.norm(achieved_goals[-1] - desired_goals[0]) * 1000))
                ani = generic_animate_trajectory(achieved_goals, desired_goals, r1,r2,r3, system, training_step=i, tol=True)
            else:
                # Run a episode reaching a goal
                achieved_goals, desired_goals, r1, r2, r3 = run_episode(env, model)
                print('Tip Error (): ' + str(np.linalg.norm(achieved_goals[-1] - desired_goals[0]) * 1000))
                ani = animate_trajectory(achieved_goals, desired_goals, r1, r2, r3, training_step=i, tol=True)
            ani.save(file_name, writer='imagemagick', fps=5)

    clips = []
    for i, file_name in enumerate(file_names):
        clip = VideoFileClip(file_name)
        clips.append(clip)

    clips_combined = concatenate_videoclips(clips)
    print("output to: " + str(output_path) + "/training_animation.mp4")
    clips_combined.write_videofile(output_path + '/training_animation.mp4')
