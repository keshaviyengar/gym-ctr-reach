import gym
import numpy as np
from ctr_reach_envs.envs.ctr_reach_env import CtrReachEnv
# rosbridge imports
import roslibpy
from roslibpy.tf import TFClient
import time

NUM_TUBES = 3


class CtrReachHardwareEnv(gym.GoalEnv):
    def __init__(self, ctr_systems_parameters, goal_tolerance_parameters, noise_parameters, joint_representation,
                 initial_joints, constrain_alpha, extension_action_limit, rotation_action_limit,
                 max_steps_per_episode, n_substeps, evaluation, select_systems, home_offset, max_retraction, max_rotation,
                 resample_joints=True, length_based_sample=False, domain_rand=0.0):
        # ROS initialization and bridge setup
        self.ros_client = roslibpy.Ros(host='localhost', port=9090)
        self.ros_client.on_ready(lambda: print('Is ROS connected status: ', self.ros_client.is_connected))
        self.read_joints_service = roslibpy.Service(self.ros_client, '/read_joint_states', 'std_srvs/Trigger')
        self.joint_command_pub = roslibpy.Topic(self.ros_client, '/joint_command', 'sensor_msgs/JointState')
        self.joint_command_pub.advertise()
        # Setup a transform listener
        self.tf_client = TFClient(self.ros_client, fixed_frame='/entry_point')
        self.tf_client.subscribe("/aurora_marker1", self.update_marker_tf)
        # Setup a publisher for the desired goal position
        self.desired_goal_pub = roslibpy.Topic(self.ros_client, '/desired_goal', 'geometry_msgs/PointStamped')
        self.desired_goal_pub.advertise()
        self.ros_client.run()

        # Create Reach Environment
        self.env = CtrReachEnv(ctr_systems_parameters, goal_tolerance_parameters, noise_parameters,
                               joint_representation, initial_joints, constrain_alpha, extension_action_limit,
                               rotation_action_limit, max_steps_per_episode, n_substeps, evaluation, select_systems,
                               home_offset, max_retraction, max_rotation, resample_joints, length_based_sample, domain_rand)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.achieved_goal = None
        self.desired_goal = None
        self.marker_pose = None

    def reset(self, goal=None, joints=None):
        # Reset timesteps
        self.achieved_goal = self.marker_pose[:3]
        if goal is not None:
            obs = self.env.reset(goal=goal)
        else:
            obs = self.env.reset()
        if joints is not None:
            self.env.trig_obj.joints = joints
        dg_msg = roslibpy.Message({'header': roslibpy.Header(frame_id='/entry_point', stamp=roslibpy.Time.now(), seq=1),
                                   'point': {'x': self.env.desired_goal[0], 'y': self.env.desired_goal[1],
                                             'z': self.env.desired_goal[2]}})
        if self.ros_client.is_connected:
            self.desired_goal_pub.publish(dg_msg)
        obs['achieved_goal'] = self.achieved_goal
        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        joints = self.env.trig_obj.joints.copy()
        joints = np.clip(joints, self.env.trig_obj.joint_spaces[0].low,
                         self.env.trig_obj.joint_spaces[0].high)
        joints[:3] = joints[:3] * 1000.0
        joints[3:] = np.rad2deg(joints[3:])
        self.joint_command_pub.publish(
            roslibpy.Message({'header': roslibpy.Header(frame_id='', stamp=roslibpy.Time.now(), seq=1),
                              'position': list(joints)}))

        dg_msg = roslibpy.Message({'header': roslibpy.Header(frame_id='/entry_point', stamp=roslibpy.Time.now(), seq=1),
                                   'point': {'x': self.env.desired_goal[0], 'y': self.env.desired_goal[1],
                                             'z': self.env.desired_goal[2]}})
        if self.ros_client.is_connected:
            self.desired_goal_pub.publish(dg_msg)
        time.sleep(0.5)
        # Publish new joints and wait
        self.achieved_goal = self.marker_pose[:3]

        reward = self.compute_reward(self.achieved_goal, self.env.desired_goal, dict())
        done = (reward == 0) or (self.env.t >= self.env.max_steps_per_episode)
        obs['achieved_goal'] = self.achieved_goal
        d = np.linalg.norm(self.achieved_goal - self.env.desired_goal, axis=-1)
        info['error'] = d
        info['joints'] = joints
        info['achieved_goal'] = self.achieved_goal
        info['desired_goal'] = self.env.desired_goal

        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode='', **kwargs):
        self.env.render(mode, **kwargs)

    def update_marker_tf(self, msg):
        self.marker_pose = np.array([msg['translation']['x'],
                                     msg['translation']['y'],
                                     msg['translation']['z'],
                                     msg['rotation']['x'],
                                     msg['rotation']['y'],
                                     msg['rotation']['z'],
                                     msg['rotation']['w'],
                                     ])

    def update_goal_tolerance(self, timestep):
        self.env.update_goal_tolerance(timestep)

    def print_parameters(self):
        self.env.print_parameters()

    def get_goal_tolerance(self):
        return self.env.goal_tolerance.get_tol()


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
    model_path = '/home/keshav/catkin_ws/src/ctr/ctr_policy_ros/example_model/tro_constrain_3/her/' \
                 'CTR-Reach-v0_1/rl_model_3000000_steps.zip'
    env_kwargs = {'resample_joints': False, 'initial_joints': np.array([-97.0e-3, -50.0e-3, -22.0e-3]),
                  'goal_tolerance_parameters': {
            'inc_tol_obs': True, 'final_tol': 0.001, 'initial_tol': 0.020,
            'N_ts': 200000, 'function': 'constant', 'set_tol': 0.001
        }
    }
    env, model = load_agent("CTR-Reach-Hardware-v0", env_kwargs, model_path)
    # env, model = load_agent("CTR-Reach-v0", env_kwargs, model_path)
    obs = env.reset()
    time.sleep(0.5)
    for i in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print('step: ' + str(i) + ' error: ' + str(info['error'] * 1000))
        time.sleep(0.1)
        if done:
            print("Done...")
            break
