import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''
This class renders the current state of the robot system using matplotlib in 3d. Called by the render function in
ctr_reach_env.py.
'''


class Ctr3dGraph(object):
    def __init__(self):
        # Create a figure on screen and set the title
        fig = plt.figure()
        # Show the graph without blocking the rest of the program
        plt.show(block=False)
        self.ax3d = plt.axes(projection='3d')

    def render(self, current_step, achieved_goal, desired_goal, r1, r2 ,r3):
        # Plot the tubes with different colors
        self.ax3d.plot3D(r1[:, 0] * 1000, r1[:, 1] * 1000, r1[:, 2] * 1000, linewidth=2.0, c='#2596BE')
        self.ax3d.plot3D(r2[:, 0] * 1000, r2[:, 1] * 1000, r2[:, 2] * 1000, linewidth=3.0, c='#D62728')
        self.ax3d.plot3D(r3[:, 0] * 1000, r3[:, 1] * 1000, r3[:, 2] * 1000, linewidth=4.0, c='#2Ca02C')
        ag = np.array(achieved_goal) * 1000
        dg = np.array(desired_goal) * 1000
        # Plot achieved and desired goal
        self.ax3d.scatter(ag[0], ag[1], ag[2], c='black', linewidth=10.0)
        self.ax3d.scatter(dg[0], dg[1], dg[2], c='magenta', linewidth=10.0)
        # Set axis limits
        self.ax3d.set_xlabel("X (mm)")
        self.ax3d.set_ylabel("Y (mm)")
        self.ax3d.set_zlabel("Z (mm)")
        self.ax3d.set_xlim3d([-100, 100])
        self.ax3d.set_xticks([-100, -50, 0, 50, 100])
        self.ax3d.set_ylim3d([-100, 100])
        self.ax3d.set_yticks([-100, -50, 0, 50, 100])
        self.ax3d.set_zlim3d([0.0, 250])
        self.ax3d.set_zticks([0, 50, 100, 150, 200, 250])
        self.ax3d.set_box_aspect([1, 1, 1])
        
        # Necessary to view frames before they are unrendered
        plt.pause(0.001)
        self.ax3d.clear()

    def close(self):
        plt.close()