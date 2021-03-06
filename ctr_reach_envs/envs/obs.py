import numpy as np
import gym

from ctr_reach_envs.envs.goal_tolerance import GoalTolerance
from ctr_reach_envs.envs.obs_utils import *

'''
The Obs class defines the observation space and defines functions to create a obs given joint values, desired and
achieved goals. Also sets actions with constraints to the system.
'''

NUM_TUBES = 3
# Extension tolerance TODO: Maybe not needed
EXT_TOL = 1e-3
# Prevent reaching maximum retraction position with causes issues in model
ZERO_TOL = 1e-4

class Obs(object):
    def __init__(self, system_parameters, goal_tolerance_parameters, noise_parameters, initial_joints,
                 joint_representation, constrain_alpha=False):
        self.system_parameters = system_parameters
        self.num_systems = len(self.system_parameters)
        # Get tube lengths to set maximum beta value in observation space
        self.tube_lengths = list()
        for sys_param in self.system_parameters:
            tube_length = list()
            for tube in sys_param:
                tube_length.append(tube.L)
            self.tube_lengths.append(tube_length)

        # Goal tolerance parameters
        self.goal_tolerance = GoalTolerance(goal_tolerance_parameters)

        # Noise parameters used for noise induced simulations
        extension_std_noise = np.full(NUM_TUBES, noise_parameters['extension_std'])
        rotation_std_noise = np.full(NUM_TUBES, noise_parameters['rotation_std'])
        self.q_std_noise = np.concatenate((extension_std_noise, rotation_std_noise))
        self.tracking_std_noise = np.full(3, noise_parameters['tracking_std'])

        self.constrain_alpha = constrain_alpha

        # Variables for joint values and relative joint values
        self.joints = initial_joints
        self.joint_representation = joint_representation

        self.joint_spaces, self.joint_sample_spaces = self.get_joint_space()

        self.observation_space = self.get_observation_space()

    def get_joint_space(self):
        """
        Get the joint space to constrain rotation and extension to limits. Seperately, create joint_sample_space from
        which samples are taken to get desired goals in the robot workspace.
        :return: The joint space and sample joint space.
        """
        joint_spaces = list()
        joint_sample_spaces = list()
        for tube_betas in self.tube_lengths:
            joint_sample_spaces.append(gym.spaces.Box(low=np.concatenate((-np.array(tube_betas) + EXT_TOL,
                                                                          np.full(NUM_TUBES, -np.pi))),
                                                      high=np.concatenate((np.full(NUM_TUBES, 0),
                                                                           np.full(NUM_TUBES, np.pi)))
                                                      ))
            if self.constrain_alpha:
                joint_spaces.append(gym.spaces.Box(low=np.concatenate((-np.array(tube_betas) + EXT_TOL,
                                                                       np.full(NUM_TUBES, -np.pi))),
                                                   high=np.concatenate((np.full(NUM_TUBES, 0),
                                                                        np.full(NUM_TUBES, np.pi)))
                                                   ))
            else:
                joint_spaces.append(gym.spaces.Box(low=np.concatenate((-np.array(tube_betas) + EXT_TOL,
                                                                       np.full(NUM_TUBES, -np.inf))),
                                                   high=np.concatenate((np.full(NUM_TUBES, 0),
                                                                        np.full(NUM_TUBES, np.inf)))
                                                   ))
        return joint_spaces, joint_sample_spaces

    def get_rep_space(self):
        """
        Get the trigonometric representation space.
        :return: rep_space that defines the trigonometric representation space.
        """
        rep_low = np.array([])
        rep_high = np.array([])
        max_tube_lengths = np.amax(np.array(self.tube_lengths), axis=0)
        for tube_length in max_tube_lengths:
            rep_low = np.append(rep_low, [-1, -1, -tube_length + ZERO_TOL])
            rep_high = np.append(rep_high, [1, 1, 0])
        rep_space = gym.spaces.Box(low=rep_low, high=rep_high, dtype="float32")
        return rep_space

    def get_observation_space(self):
        """
        Get the observation space defining the limits of the obs as defined in gym
        :return: The observation space
        """
        initial_tol = self.goal_tolerance.init_tol
        final_tol = self.goal_tolerance.final_tol
        rep_space = self.get_rep_space()

        # Set overall goal error min and max
        # Not actually used to constrain observation, just used for definition
        del_x_y_min = -0.2
        del_x_y_max = 0.2
        del_z_min = -0.2
        del_z_max = 0.2
        #del_x_y_min = -0.5
        #del_x_y_max = 0.5
        #del_z_min = 0.0
        #del_z_max = 0.5

        # If training a single system, don, include the psi variable indicating system
        if self.num_systems == 1:
            obs_space_low = np.concatenate(
                (rep_space.low, np.array([del_x_y_min, del_x_y_min, del_z_min, final_tol])))
            obs_space_high = np.concatenate(
                (rep_space.high, np.array([del_x_y_max, del_x_y_max, del_z_max, initial_tol])))

        else:
            obs_space_low = np.concatenate(
                (rep_space.low, np.array([del_x_y_min, del_x_y_min, del_z_min, final_tol, 0])))
            obs_space_high = np.concatenate(
                (rep_space.high, np.array([del_x_y_max, del_x_y_max, del_z_max, initial_tol, self.num_systems - 1])))
        observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(low=np.array([-0.1, -0.1, 0]),
                                        high=np.array([0.1, 0.1, 0.2]), dtype="float32"),
            achieved_goal=gym.spaces.Box(low=np.array([-0.1, -0.1, 0]),
                                         high=np.array([0.1, 0.1, 0.2]), dtype="float32"),
            observation=gym.spaces.Box(
                low=obs_space_low,
                high=obs_space_high,
                dtype="float32")
        ))
        return observation_space

    def get_obs(self, desired_goal, achieved_goal, goal_tolerance, system):
        """
        The an observation object given the current desired, achieved goals, goal tolerance and system selected
        :param desired_goal: Current desired goal
        :param achieved_goal: Current achieved goal of the end-effector
        :param goal_tolerance: Current goal tolerance
        :param system: Selected system
        :return: Observation object
        """
        # Get trigonometric representation based on if egocentric on proprioceptive
        assert self.joint_representation in ['egocentric', 'proprioceptive'], "Incorrect joint representation selected."
        if self.joint_representation == 'egocentric':
            rel_joints = prop2ego(self.joints)
            trig_joints = joint2rep(rel_joints)
        else:
            trig_joints = joint2rep(self.joints)

        if self.num_systems > 1:
            obs = np.concatenate([trig_joints, desired_goal - achieved_goal, np.array([goal_tolerance, system])])
        else:
            obs = np.concatenate([trig_joints, desired_goal - achieved_goal, np.array([goal_tolerance])])

        self.obs = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': desired_goal.copy()
        }

        return self.obs

    def set_action(self, action, system):
        """
        Set the given action to the joints for the selected system.
        :param action: Array of rotation and extensions for each tube.
        :param system: The selected system.
        """
        self.joints = np.clip(self.joints + action, self.joint_spaces[system].low, self.joint_spaces[system].high)
        betas = self.joints[:NUM_TUBES]
        alphas = self.joints[NUM_TUBES:]
        # Apply extension joint constraints, rotation constraints applied through joint_spaces.
        for i in range(1, NUM_TUBES):
            # Ordering is reversed, since we have innermost as last whereas in constraints its first.
            # Bi-1 <= Bi
            # Bi-1 >= Bi - Li-1 + Li
            betas[i - 1] = min(betas[i - 1], betas[i])
            betas[i - 1] = max(betas[i - 1], self.tube_lengths[system][i] - self.tube_lengths[system][i - 1] + betas[i])

        self.joints = np.concatenate((betas, alphas))

    def sample_goal(self, system):
        """
        Sample a joint goal while considering constraints on extension and joint limits.
        :param system: The system to to sample the goal.
        :return: Constrained achievable joint values.
        """
        counter = 0
        while True:
            joint_sample = self.joint_sample_spaces[system].sample()
            betas = joint_sample[:NUM_TUBES]
            alphas = joint_sample[NUM_TUBES:]
            # Apply constraints
            valid_joint = []
            for i in range(1, NUM_TUBES):
                valid_joint.append((betas[i - 1] <= betas[i]) and (
                        betas[i - 1] + self.tube_lengths[system][i - 1] >= self.tube_lengths[system][i] + betas[i]))
            counter += 1
            if all(valid_joint):
                break
            if counter > 1000:
                raise ValueError("Stuck sampling goals...")
        joint_constrain = np.concatenate((betas, alphas))
        return joint_constrain

