import numpy as np
from ctr_reach_envs.envs.CTR_Python import Tube


def sample_parameters(tube_parameters, randomization):
    """
    Given a set of tube parameters, randomize the parameters within a percentage interval.
    :param tube_parameters: System parameters like length, curved length, diameters etc.
    :param randomization: The percentage interval to sample from.
    :return: Sampled parameters based on randomization.
    """
    # Do not randomize length as causes issues with extension constraints
    L = randomize_value(tube_parameters.L, 0)
    L_c = randomize_value(tube_parameters.L_c, randomization)
    diameter_inner = randomize_value(tube_parameters.diameter_inner, randomization)
    diameter_outer = randomize_value(tube_parameters.diameter_outer, randomization)
    stiffness = randomize_value(tube_parameters.E, randomization)
    torsional_stiffness = randomize_value(tube_parameters.G, randomization)
    x_curvature = randomize_value(tube_parameters.U_x, randomization)
    y_curvature = randomize_value(tube_parameters.U_y, 0)

    new_parameters = Tube(L, L_c, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature,
                          y_curvature)
    return new_parameters


def randomize_value(value, randomization):
    """
    Give a value, take a sample based on randomization value that specifies interval range.
    :param value: Parameter value.
    :param randomization: Randomization percentage that specifies interval range to sample.
    :return: New sampled value.
    """
    sampled_value = np.random.uniform(value - value * randomization, value + value * randomization)
    return sampled_value
