import gym
import numpy as np
from ctr_reach_envs.envs.obs import Obs
from ctr_reach_envs.envs.goal_tolerance import GoalTolerance
from ctr_reach_envs.envs.model import Model
from ctr_reach_envs.envs.ctr_3d_graph import Ctr3dGraph

from ctr_reach_envs.envs.CTR_Python import Tube

NUM_TUBES = 3


class CtrReachEnv(gym.GoalEnv):
    def __init__(self, ctr_systems_parameters, goal_tolerance_parameters, noise_parameters, joint_representation,
                 initial_joints, constrain_alpha, extension_action_limit, rotation_action_limit, max_steps_per_episode,
                 n_substeps, evaluation, select_systems, home_offset, max_retraction, max_rotation,
                 resample_joints=True,
                 length_based_sample=False, domain_rand=0.0):

        # Load in all system parameters
        self.ctr_system_parameters = list()
        for system in ctr_systems_parameters:
            tubes = list()
            for tube in ctr_systems_parameters[system]:
                tubes.append(Tube(**ctr_systems_parameters[system][tube]))
            self.ctr_system_parameters.append(tubes)

        # Remove unused systems based of select_systems
        self.select_systems = select_systems
        self.ctr_system_parameters = [self.ctr_system_parameters[system] for system in select_systems]
        self.noise_parameters = noise_parameters
        assert joint_representation in ['egocentric', 'proprioceptive']
        self.joint_representation = joint_representation
        self.max_steps_per_episode = max_steps_per_episode
        self.n_substeps = n_substeps
        # Other parameters and settings
        self.starting_joints = initial_joints
        self.desired_joints = initial_joints
        self.constrain_alpha = constrain_alpha
        self.evaluation = evaluation
        self.resample_joints = resample_joints
        self.length_based_sample = length_based_sample
        self.domain_rand = domain_rand

        # Home offset for hardware tubes being very long for carriages. To account for twisting that occurs.
        self.home_offset = home_offset
        self.max_retraction = max_retraction

        # CTR kinematic model
        self.model = Model(self.ctr_system_parameters)

        self.visualization = None

        # Initialization parameters / objects
        self.t = 0
        self.trig_obj = Obs(self.ctr_system_parameters, goal_tolerance_parameters, noise_parameters, initial_joints,
                            joint_representation, home_offset, max_retraction, max_rotation, constrain_alpha)
        self.observation_space = self.trig_obj.get_observation_space()

        self.extension_action_limit = extension_action_limit
        self.rotation_action_limit = rotation_action_limit
        # Action space definition
        beta_action = np.full(NUM_TUBES, self.extension_action_limit)
        alpha_action = np.full(NUM_TUBES, np.deg2rad(self.rotation_action_limit))
        self.action_space = gym.spaces.Box(low=np.concatenate((-beta_action, -alpha_action)),
                                           high=np.concatenate((beta_action, alpha_action)),
                                           dtype="float32")
        self.system = 0
        # Initialization of starting position
        self.starting_position = self.model.forward_kinematics(np.array(self.starting_joints), self.system)
        self.desired_goal = self.starting_position
        # Goal tolerance parameters
        self.goal_tolerance = GoalTolerance(goal_tolerance_parameters)

    def reset(self, goal=None, system=None):
        """
        Reset function as specified by gym. Called at the start of each episode.
        :param goal: Give the agent a specific goal to reach. Set to None if sample a new desired goal.
        :param system: The CTR system to use for this episode.
        :return: The observation or achieved, desired goals and joints in trigonometric representation
        """
        # Reset timesteps
        self.t = 0
        # Domain randomization for domain transfer. Set self.domain_rand to zero if not used
        self.model.randomize_parameters(self.domain_rand)
        # Set system to None if training and want to sample systems at each episode
        if system is None:
            # Use non-uniform sampling based on length of each system
            if self.length_based_sample:
                # Get overall lengths of systems
                all_system_length = 0
                system_length = []
                for system in self.system_parameters:
                    all_system_length += system[0].L
                    system_length.append(system[0].L)
                sys_prob = np.array(system_length) / all_system_length
                self.system = np.where(np.random.multinomial(1, sys_prob) == 1)[0][0]
            else:
                # Sample uniformly
                self.system = np.random.randint(len(self.ctr_system_parameters))
        else:
            self.system = system
        if goal is None:
            # No goal given so sample a desired goal in the robot workspace
            self.desired_joints = self.trig_obj.sample_goal(self.system)
            self.desired_goal = self.model.forward_kinematics(self.desired_joints, self.system)
        else:
            self.desired_goal = goal
        if self.resample_joints:
            # Should the initial position of the robot be re-sampled at the start of each episode
            self.starting_joints = self.trig_obj.sample_goal(self.system)
            self.trig_obj.joints = self.starting_joints
            self.starting_position = self.model.forward_kinematics(self.trig_obj.joints, self.system)
        else:
            # Start from the final position of last episode
            self.starting_joints = self.trig_obj.joints
            self.starting_position = self.model.forward_kinematics(self.starting_joints, self.system)

        return self.trig_obj.get_obs(self.desired_goal, self.starting_position, self.goal_tolerance.get_tol(),
                                     self.system)

    def seed(self, seed=None):
        """
        Set the seed
        :param seed: Value of seed. If None, sample one.
        """
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        """
        Step function as defined by gym. Takes input of action and simulates the environment by one step.
        :param action: Selected action or changes in joint values.
        :return: New observation of desired, achieved goals and trigonometric joint representation
        """
        # Ensure actions are not NaNs and within action space to enforce constraints
        assert not np.all(np.isnan(action))
        assert self.action_space.contains(action)
        # For n_substeps, repeat the selected action
        for _ in range(self.n_substeps):
            self.trig_obj.set_action(action, self.system)
        # Compute achieved goal with forward kinematics
        achieved_goal = self.model.forward_kinematics(self.trig_obj.joints, self.system)
        self.t += 1
        reward = self.compute_reward(achieved_goal, self.desired_goal, {'robot_length': self.desired_joints[0]})
        done = (reward == 0) or (self.t >= self.max_steps_per_episode)
        obs = self.trig_obj.get_obs(self.desired_goal, achieved_goal, self.goal_tolerance.get_tol(), self.system)

        # If evaluating, save more information for analysis
        if self.evaluation:
            info = {'is_success': (np.linalg.norm(self.desired_goal - achieved_goal) < self.goal_tolerance.get_tol()),
                    'errors_pos': np.linalg.norm(self.desired_goal - achieved_goal),
                    'errors_orient': 0,
                    'system_idx': self.select_systems[self.system],
                    'position_tolerance': self.goal_tolerance.get_tol(),
                    'orientation_tolerance': 0,
                    'achieved_goal': achieved_goal,
                    'desired_goal': self.desired_goal, 'starting_position': self.starting_position,
                    'q_desired': self.desired_joints, 'q_achieved': self.trig_obj.joints,
                    'q_starting': self.starting_joints}
        else:
            if self.goal_tolerance.measure == 'percentage':
                errors = np.linalg.norm(self.desired_goal - achieved_goal) / (self.desired_goal[2])
            else:
                errors = np.linalg.norm(self.desired_goal - achieved_goal)
            info = {'is_success': ( errors < self.goal_tolerance.get_tol()),
                    'error': errors}

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward given current and desired goals based on Euclidean distance.
        :param achieved_goal: Current achieved position of end-effector.
        :param desired_goal: Desired position of end-effector.
        :param info: Dictionary for extra details.
        :return: -1 or 0 based on current tolerance.
        """
        assert achieved_goal.shape == desired_goal.shape
        if self.goal_tolerance.measure == 'percentage':
            robot_length = info['robot_length']
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1) / robot_length
            return -(d > self.goal_tolerance.get_tol()).astype(np.float64)

        else:
            d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return -(d > self.goal_tolerance.get_tol()).astype(np.float64)

    def render(self, mode='empty', **kwargs):
        """
        Render the current shape of the CTR system with achieved goal and desired goal.
        :param mode: Set the render mode. If not set, no rendering is performed.
        :param kwargs: Extra arguements if needed.
        """
        if mode == 'live':
            if self.visualization is None:
                self.visualization = Ctr3dGraph()
            self.visualization.render(self.t, self.trig_obj.obs['achieved_goal'], self.trig_obj.obs['desired_goal'],
                                      self.model.r1, self.model.r2, self.model.r3)

    def close(self):
        """
        Close gym environment.
        """
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None

    def print_parameters(self):
        """
        Print parameters used for experiment. Called from Callback function in rl-zoo from train.py script.
        """
        print("----Observation and q_space----")

        print("----Goal tolerance parameters----")

    def update_goal_tolerance(self, timestep):
        """

        :param timestep:  The current timestep to update the goal tolerance
        """
        self.goal_tolerance.update(timestep)

    def get_goal_tolerance(self):
        return self.goal_tolerance.get_tol()
