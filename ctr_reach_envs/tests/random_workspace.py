import gym
import ctr_generic_envs

from gym.wrappers import FlattenDictWrapper
import numpy as np
import matplotlib.pyplot as plt


# Run a longer rollout on some environments
def random_rollout(spec, kwargs):
    positions = np.array([])
    env = FlattenDictWrapper(spec.make(**kwargs), ['observation', 'desired_goal', 'achieved_goal'])
    agent = lambda ob: env.action_space.sample()
    ob = env.reset()
    for _ in range(10):
        assert env.observation_space.contains(ob)
        a = agent(ob)
        assert env.action_space.contains(a)
        (ob, _reward, done, _info) = env.step(a)
        if done:
            break
    env.close()

# Run a longer rollout on some environments
def random_reset(spec, num_points, kwargs):
    positions = np.zeros((3, num_points))
    env = spec.make(**kwargs)
    for i in range(num_points):
        ob = env.reset()
        positions[:, i] = ob["desired_goal"]
    return positions

def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xs = points[0, :]
    ys = points[1, :]
    zs = points[2, :]

    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    plt.show()


if __name__ == '__main__':
    spec = gym.spec('CTR-Generic-Reach-v0')
    kwargs = {"resample_joints": True}
    sampled_positions = random_reset(spec, 1000, kwargs)
    plot_points(sampled_positions)

