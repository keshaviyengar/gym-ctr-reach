import gym
import ctr_reach_envs

from gym.wrappers import FlattenDictWrapper
import numpy as np

'''
The script below runs basic tests to ensure the environment given is compatible with gym and the hindsight experience
replay wrapper. Also, runs a random rollout given random actions to test the step, reset functions of the custom
environment.
'''

def environment(spec, kwargs):
    """
    Given a environment specification and arguements, test that environment is compatible with gym and HER
    :param spec: Gym specification.
    :param kwargs: Extra arguements for environments.
    """
    env = FlattenDictWrapper(spec.make(**kwargs), ['observation', 'desired_goal', 'achieved_goal'])
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), 'Reset observation: {!r} not in space'.format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), 'Step observation: {!r} not in space'.format(observation)
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get('render.modes', []):
        env.render(mode=mode)

    env.close()


def random_rollout(spec, num_episodes, kwargs):
    """
    Run through a rollout with random actions.
    :param spec: Environment specification.
    :param kwargs: Environment extra arguements.
    """
    env = FlattenDictWrapper(spec.make(**kwargs), ['observation', 'desired_goal', 'achieved_goal'])
    agent = lambda ob: env.action_space.sample()
    for episode in range(num_episodes):
        ob = env.reset()
        for step in range(150):
            assert env.observation_space.contains(ob)
            a = agent(ob)
            assert env.action_space.contains(a)
            (ob, _reward, done, _info) = env.step(a)
            env.render(mode='live')
            if done:
                break
    env.close()


def test_environment():
    """
    Test the CTR-Reach-v0 environment and run a random rollout
    """
    spec = gym.spec('CTR-Reach-v0')
    kwargs = {}
    #environment(spec, kwargs)
    random_rollout(spec, 5, kwargs)

if __name__ == '__main__':
    test_environment()