import numpy as np

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# This class should generate a trajectory for an agent to follow.
# 1. Inputs should be size of trajectory as well as locations
# 2. Type of trajectories (square, circle, triangle, corkscrew)
# 3. A function to be called that will go through steps and return a Cartesian position

def circle_traj(num_points, centre_x, centre_y, centre_z, radius):
    t = np.linspace(0, 2*np.pi, num_points)
    x = centre_x + (np.cos(t) * radius)
    y = centre_y + (np.sin(t) * radius)
    z = np.zeros_like(t) + centre_z
    return x,y,z

def line_traj(num_points, start_x, start_y, start_z, end_x, end_y, end_z):
    x = np.linspace(start_x, end_x, num_points)
    y = np.linspace(start_y, end_y, num_points)
    z = np.linspace(start_z, end_z, num_points)
    return x,y,z

def polygon_traj(num_points, points):
    x = []
    y = []
    z = []
    for i in range(1, len(points)):
        start = points[i-1]
        end = points[i]
        x_temp, y_temp, z_temp = line_traj(num_points, start[0], start[1], start[2], end[0], end[1], end[2])
        x.append(x_temp)
        y.append(y_temp)
        z.append(z_temp)

    return np.concatenate(x), np.concatenate(y), np.concatenate(z)

def helix_traj(num_points, num_revs, R, a, xyz=[0,0,0]):
    t = np.linspace(0, num_revs * 2 * np.pi, num_points)
    x = R * np.cos(t) + xyz[0]
    y = R * np.sin(t) + xyz[1]
    z = a * t + xyz[2]
    return x,y,z
