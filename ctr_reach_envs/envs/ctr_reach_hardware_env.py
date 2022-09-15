import gym
import numpy as np
from ctr_reach_envs.envs.obs import Obs
from ctr_reach_envs.envs.model import Model
from ctr_reach_envs.envs.goal_tolerance import GoalTolerance
from CTR_Python.Tube import Tube
# rosbridge imports
import roslibpy
from roslibpy.tf import TFClient
import time

NUM_TUBES = 3


class CtrReachHardwareEnv(gym.GoalEnv):
    def __init__(self, ctr_systems_parameters, goal_tolerance_parameters, noise_parameters, joint_representation,
                 initial_joints, constrain_alpha, extension_action_limit, rotation_action_limit,
                 max_steps_per_episode, n_substeps, evaluation, select_systems, resample_joints=True,
                 length_based_sample=False, domain_rand=0.0):
        # ROS initialization and bridge setup
        self.ros_client = roslibpy.Ros(host='localhost', port=9090)
        self.ros_client.on_ready(lambda: print('Is ROS connected status: ', self.ros_client.is_connected))
        self.read_joints_service = roslibpy.Service(self.ros_client, '/read_joint_states', 'std_srvs/Trigger')
        # self.read_joints_sub = roslibpy.Topic(self.ros_client, '/joint_state', 'sensor_msgs/JointState')
        # self.read_joints_sub.subscribe(self.update_joint_state)
        self.joint_command_pub = roslibpy.Topic(self.ros_client, '/joint_command', 'sensor_msgs/JointState')
        self.joint_command_pub.advertise()
        # Setup a transform listener
        self.tf_client = TFClient(self.ros_client, fixed_frame='/base_link')
        self.tf_client.subscribe("/aurora_marker1", self.update_marker_tf)
        # Setup a publisher for the desired goal position
        self.achieved_goal_pub = roslibpy.Topic(self.ros_client, '/desired_goal', 'geometry_msgs/PointStamped')
        self.achieved_goal_pub.advertise()
        self.ros_client.run()

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

        # CTR kinematic model
        self.model = Model(self.ctr_system_parameters)

        # Initialization parameters / objects
        self.t = 0
        self.trig_obj = Obs(self.ctr_system_parameters, goal_tolerance_parameters, noise_parameters, initial_joints,
                            joint_representation, constrain_alpha)
        self.observation_space = self.trig_obj.get_observation_space()

        self.extension_action_limit = extension_action_limit
        self.rotation_action_limit = rotation_action_limit
        # Action space definition
        beta_action = np.full(NUM_TUBES, self.extension_action_limit)
        alpha_action = np.full(NUM_TUBES, np.deg2rad(self.rotation_action_limit))
        self.action_space = gym.spaces.Box(low=np.concatenate((-beta_action, -alpha_action)),
                                           high=np.concatenate((beta_action, alpha_action)),
                                           dtype="float64")
        self.system = 0
        # Initialization of starting position
        self.achieved_goal = None
        self.desired_goal = None
        self.marker_pose = None
        # Goal tolerance parameters
        self.goal_tolerance = GoalTolerance(goal_tolerance_parameters)

    def reset(self, goal=None):
        # Reset timesteps
        self.t = 0
        self.achieved_goal = self.marker_pose[:3]
        if goal is None:
            # No goal given so sample a desired goal in the robot workspace
            self.desired_joints = self.trig_obj.sample_goal(self.system)
            self.desired_joints[3:] = np.pi / 2
            self.desired_goal = self.model.forward_kinematics(self.desired_joints, self.system)
        else:
            self.desired_goal = goal
        dg_msg = roslibpy.Message({'header': roslibpy.Header(frame_id='/base_link', stamp=roslibpy.Time.now(), seq=1),
                                   'point': {'x': self.desired_goal[0], 'y': self.desired_goal[1],
                                             'z': self.desired_goal[2]}})
        if self.ros_client.is_connected:
            self.achieved_goal_pub.publish(dg_msg)
        return self.trig_obj.get_obs(self.desired_goal, self.achieved_goal, self.goal_tolerance.get_tol(), self.system)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        # Ensure actions are not NaNs and within action space to enforce constraints
        assert not np.all(np.isnan(action))
        assert self.action_space.contains(action)
        # For n_substeps, repeat the selected action
        for _ in range(self.n_substeps):
            self.trig_obj.set_action(action, self.system)
        self.joint_command_pub.publish(
            roslibpy.Message({'header': roslibpy.Header(frame_id='', stamp=roslibpy.Time.now(), seq=1),
                              'position': list(self.trig_obj.joints * 1000.0)}))
        time.sleep(2.0)
        # Publish new joints and wait
        achieved_goal = self.marker_pose[:3]

        reward = self.compute_reward(achieved_goal, self.desired_goal, dict())
        done = (reward == 0) or (self.t >= self.max_steps_per_episode)
        obs = self.trig_obj.get_obs(self.desired_goal, achieved_goal, self.goal_tolerance.get_tol(), self.system)

        info = {}
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert achieved_goal.shape == desired_goal.shape
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.goal_tolerance.get_tol()).astype(np.float64)

    def render(self, mode='text', **kwargs):
        # Publish ROS logs
        pass

    def update_marker_tf(self, msg):
        self.marker_pose = np.array([msg['translation']['x'],
                                     msg['translation']['y'],
                                     msg['translation']['z'],
                                     msg['rotation']['x'],
                                     msg['rotation']['y'],
                                     msg['rotation']['z'],
                                     msg['rotation']['w'],
                                     ])

    def update_joint_state(self, msg):
        # TODO is this needed?
        self.joint_state = np.array(msg['position'])

    def update_goal_tolerance(self, timestep):
        self.goal_tolerance.update(timestep)


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


if __name__ == '__main__':
    model_path = '/home/keshav/catkin_ws/src/ctr/ctr_policy_ros/example_model/rvim_hardware/her/' \
                 'CTR-Reach-v0_1/rl_model_3000000_steps.zip'
    env_kwargs = {
        'goal_tolerance_parameters': {
            'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0.005
        }
    }
    env, model = load_agent("CTR-Reach-Hardware-v0", env_kwargs, model_path)
    time.sleep(1.0)
    # TODO: Load in policy
    obs = env.reset()
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        action[3:] = 0
        # TODO: Weird thing where can't be equal to limits
        action[action < env.action_space.low] += 1e-9
        action[action > env.action_space.high] -= 1e-9
        obs, reward, done, info = env.step(action)
        time.sleep(0.1)
        if done:
            print("reached goal!")
            break
