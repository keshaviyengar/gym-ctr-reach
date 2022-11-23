from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import numpy as np
import pandas as pp

from ctr_reach_envs.envs.CTR_Python.inverse_kinematics import JacobianIk
from ctr_reach_envs.src.plotting_utils import plot_path_only, compare_paths_only
from paths import line_traj, circle_traj, polygon_traj, helix_traj, velocity_based_line_traj, velocity_based_circle_traj
from ctr_reach_envs.envs.CTR_Python.Tube import Tube
from ctr_reach_envs.envs.CTR_Python.CTR_Model import CTR_Model

from ctr_reach_envs.src.utils import load_agent, run_episode, trajectory_controller


def get_intial_starting_q(x_0):
    # Use RL to finding starting position of first point
    gen_model_path = "/her/CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0.zip"
    project_folder = '/home/keshav/ctm2-stable-baselines/gym-ctr-reach/ctr_reach_envs/saved_policies/'
    name = 'rotation_experiments/free_rotation/tro_free_0'
    selected_systems = [0]
    model_path = project_folder + name + gen_model_path
    env_id = "CTR-Reach-v0"
    env_kwargs = {'evaluation': True, 'joint_representation': 'egocentric', 'resample_joints': False, 'constrain_alpha': False,
                  'select_systems': selected_systems,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001},
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)

    ags, dgs, q, _, _, _ = run_episode(env, model, goal=x_0)
    return q

def path_following_RL(x_points, y_points, z_points):
    x_0 = np.array([x_points[0], y_points[0], z_points[0]])
    q = get_intial_starting_q(x_0)
    gen_model_path = "/her/CTR-Generic-Reach-v0_1/CTR-Generic-Reach-v0.zip"

    project_folder = '/home/keshav/ctm2-stable-baselines/gym-ctr-reach/ctr_reach_envs/saved_policies/'
    name = 'rotation_experiments/free_rotation/tro_free_0'
    selected_systems = [0]
    model_path = project_folder + name + gen_model_path
    env_id = "CTR-Reach-v0"
    env_kwargs = {'evaluation': True, 'joint_representation': 'egocentric', 'resample_joints': False, 'constrain_alpha': False,
                  'select_systems': selected_systems, 'initial_joints': q,
                  'goal_tolerance_parameters': {'inc_tol_obs': True, 'initial_tol': 0.020, 'final_tol': 0.001,
                                                'N_ts': 200000, 'function': 'constant', 'set_tol': 0.0005},
                  }
    env, model = load_agent(env_id, env_kwargs, model_path)

    # Run through trajectory controller and save goals and shape
    achieved_goals, desired_goals, r1, r2, r3 = trajectory_controller(model, env, x_points, y_points, z_points, 0, selected_systems)
    return achieved_goals, desired_goals


if __name__ == '__main__':
    # Defining parameters of each tube, numbering starts with the most inner tube
    # length, length_curved, diameter_inner, diameter_outer, stifness, torsional_stiffness, x_curvature, y_curvature
    tube1 = Tube(431e-3, 103e-3, 2 * 0.35e-3, 2 * 0.55e-3, 10.25e+10, 18.79e+10, 21.3, 0)
    tube2 = Tube(332e-3, 113e-3, 2 * 0.7e-3, 2 * 0.9e-3, 68.6e+10, 11.53e+10, 13.1, 0)
    tube3 = Tube(174e-3, 134e-3, 2e-3, 2 * 1.1e-3, 16.96e+10, 14.25e+10, 3.5, 0)
    # initial twist (for ivp solver)
    uz_0 = np.array([0.0, 0.0, 0.0])
    u1_xy_0 = np.array([[0.0], [0.0]])
    f = np.array([0, 0, 0]).reshape(3, 1)

    # Get starting joint position of path with RL
    #q = get_intial_starting_q(path_array[0, :])
    q = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
    # Initial position of joints
    q_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    CTR = CTR_Model(tube1, tube2, tube3, f, q, q_0, 0.01, 1)
    J = CTR.jac(np.concatenate((u1_xy_0, uz_0), axis=None))
    x_d = CTR.r[-1, :]

    path_type = 'circle'

    if path_type == 'line':
        #x_points, y_points, z_points = line_traj(20, -0.1, 0.05, 0.20, 0.1, 0.05, 0.20)
        v = [0.001, 0.001, 0]
        x_points, y_points, z_points = velocity_based_line_traj(10, 10, v, x_d[0], x_d[1], x_d[2])
    if path_type == 'helix':
        x_points, y_points, z_points = helix_traj(100, 3, 0.03, 0.005, [0.06, 0.06, 0.18])
    if path_type == 'circle':
        x_points, y_points, z_points = velocity_based_circle_traj(50, 30, 0.10, 0.01, x_d[0], x_d[1], x_d[2])
        #x_points, y_points, z_points = circle_traj(40, 0.0, 0.10, 0.20, 0.05)

    # Concatenate into valid path for Jacobian solver
    path_array = np.vstack((x_points, y_points, z_points)).T

    # Control parameters
    K_p = 2
    damping_constant = 0.45
    jacobian_ik = JacobianIk(tube1, tube2, tube3, K_p, damping_constant, True)
    # Path array is (n,3) matrix
    x_d_array, x_c_array, q_array = jacobian_ik.path_following(path_array, q, q_0, uz_0, u1_xy_0)

    achieved_goals, desired_goals = path_following_RL(x_d_array[:,0], x_d_array[:,1], x_d_array[:,2])

    # Plot path following
    compare_paths_only(x_c_array, achieved_goals, desired_goals)

    # Print error metrics
    print('Jacobian errors: ')
    print(str(np.mean(np.linalg.norm((x_c_array - x_d_array) * 1000, axis=1))))
    print(str(np.std(np.linalg.norm((x_c_array - x_d_array) * 1000, axis=1))))
    print('DeepRL errors: ')
    print(str(np.mean(np.linalg.norm((np.array(achieved_goals) - np.array(desired_goals)) * 1000, axis=1))))
    print(str(np.std(np.linalg.norm((np.array(achieved_goals) - np.array(desired_goals)) * 1000, axis=1))))

