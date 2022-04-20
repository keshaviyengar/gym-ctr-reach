from gym.envs.registration import register
import numpy as np

register(
    id='CTR-Reach-v0', entry_point='ctr_reach_envs.envs:CtrReachEnv',
    kwargs={
        'ctr_systems_parameters': {
            # Autonomous steering by Mohsen Khadem
            'ctr_0': {
                'tube_0':
                    {'length': 431e-3, 'length_curved': 103e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 1.10e-3,
                     'stiffness': 10.25e+10, 'torsional_stiffness': 18.79e+10, 'x_curvature': 21.3, 'y_curvature': 0
                     },

                'tube_1':
                    {'length': 332e-3, 'length_curved': 113e-3, 'diameter_inner': 1.4e-3, 'diameter_outer': 1.8e-3,
                     'stiffness': 68.6e+10, 'torsional_stiffness': 11.53e+10, 'x_curvature': 13.1, 'y_curvature': 0
                     },

                'tube_2':
                    {'length': 174e-3, 'length_curved': 134e-3, 'diameter_inner': 2e-3, 'diameter_outer': 2.4e-3,
                     'stiffness': 16.96e+10, 'torsional_stiffness': 14.25e+10, 'x_curvature': 3.5, 'y_curvature': 0
                     }
            },
            # Learning the FK and IK of a 6-DOF CTR by Grassmann
            'ctr_1': {
                'tube_0':
                    {'length': 370e-3, 'length_curved': 45e-3, 'diameter_inner': 0.3e-3, 'diameter_outer': 0.4e-3,
                     'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.8, 'y_curvature': 0,
                     },

                'tube_1':
                    {'length': 305e-3, 'length_curved':100e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 0.9e-3,
                     'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 9.27, 'y_curvature': 0,
                     },

                'tube_2':
                    {'length': 170e-3, 'length_curved': 100e-3, 'diameter_inner': 1.2e-3, 'diameter_outer': 1.5e-3,
                     'stiffness': 50e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 4.37, 'y_curvature': 0,
                     }
            },
            # RViM lab tube parameters
            'ctr_2': {
                'tube_0':
                    {'length': 309e-3, 'length_curved': 145e-3, 'diameter_inner': 0.7e-3, 'diameter_outer': 1.1e-3,
                     'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 13.52, 'y_curvature': 0
                     },
                'tube_1':
                    {'length': 275e-3, 'length_curved': 114e-3, 'diameter_inner': 1.4e-3, 'diameter_outer': 1.8e-3,
                    'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 11.68, 'y_curvature': 0
                     },
               'tube_2':
                   {'length': 173e-3, 'length_curved': 173e-3, 'diameter_inner': 1.83e-3, 'diameter_outer': 2.39e-3,
                    'stiffness': 75e+9, 'torsional_stiffness': 25e+9, 'x_curvature': 10.8, 'y_curvature': 0
                    }
            },
            # Unknown tube parameters or where they are from
            'ctr_3': {
                'tube_0':
                    {'length': 150e-3, 'length_curved': 100e-3, 'diameter_inner': 1.0e-3, 'diameter_outer': 2.4e-3,
                     'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 15.82, 'y_curvature': 0},

                'tube_1':
                    {'length': 100e-3, 'length_curved': 21.6e-3, 'diameter_inner': 3.0e-3, 'diameter_outer': 3.8e-3,
                     'stiffness': 5e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 11.8, 'y_curvature': 0},

                'tube_2':
                    {'length': 70e-3, 'length_curved': 8.8e-3, 'diameter_inner': 4.4e-3, 'diameter_outer': 5.4e-3,
                     'stiffness': 5.0e+10, 'torsional_stiffness': 2.3e+10, 'x_curvature': 20.04, 'y_curvature': 0}
            },
        },
        'extension_action_limit': 0.001,
        'rotation_action_limit': 5,
        'max_steps_per_episode': 150,
        'n_substeps': 10,
        'goal_tolerance_parameters': {
            'inc_tol_obs': False, 'final_tol': 0.001, 'initial_tol': 0.020,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0
        },
        'noise_parameters': {
            # 0.001 is the gear ratio
            # 0.001 is also the tracking std deviation for now for testing.
            'rotation_std': np.deg2rad(0), 'extension_std': 0.001 * np.deg2rad(0), 'tracking_std': 0.0
        },
        'select_systems': [0],
        'constrain_alpha': False,
        # Format is [beta_0, beta_1, ..., beta_n, alpha_0, ..., alpha_n]
        'initial_joints': np.array([0, 0, 0, 0, 0, 0]),
        'joint_representation': 'egocentric',
        'resample_joints': True,
        'evaluation': False,
        'length_based_sample': False,
        'domain_rand': 0.0
    },
    max_episode_steps=150
)
