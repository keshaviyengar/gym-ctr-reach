import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.markers import MarkerStyle
from ctr_reach_envs.src.utils import decay_goal_tolerance
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10

# Utility functions for plotting
def plot_trajectory(achived_goals, desired_goals, r1, r2, r3, save_path=None):
    fig = plt.figure(figsize=(6, 6), dpi=150)
    ax = plt.axes(projection='3d')
    ax.plot3D(r1[0][:,0] * 1000, r1[0][:,1] * 1000, r1[0][:,2] * 1000, linewidth=2.0, c='#2596BE')
    ax.plot3D(r2[0][:,0] * 1000, r2[0][:,1] * 1000, r2[0][:,2] * 1000, linewidth=3.0, c='#D62728')
    ax.plot3D(r3[0][:,0] * 1000, r3[0][:,1] * 1000, r3[0][:,2] * 1000, linewidth=4.0, c='#2Ca02C')
    ag = np.array(achived_goals) * 1000
    dg = np.array(desired_goals) * 1000
    ax.plot3D(ag[:,0], ag[:,1], ag[:,2], marker='.', linestyle=':', label='achieved', c='black')
    ax.scatter(ag[0,0], ag[0,1], ag[0,2], c='black', linewidth=10.0)
    ax.scatter(dg[0,0], dg[0,1], dg[0,2], c='magenta', linewidth=10.0)
    ax.plot3D(r1[-1][:,0] * 1000, r1[-1][:,1] * 1000, r1[-1][:,2] * 1000, linewidth=2.0, c='#2596BE')
    ax.plot3D(r2[-1][:,0] * 1000, r2[-1][:,1] * 1000, r2[-1][:,2] * 1000, linewidth=3.0, c='#D62728')
    ax.plot3D(r3[-1][:,0] * 1000, r3[-1][:,1] * 1000, r3[-1][:,2] * 1000, linewidth=4.0, c='#2CA02C')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim3d([-100, 100])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_ylim3d([-100, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_zlim3d([0.0, 250])
    ax.set_zticks([0, 50, 100, 150, 200, 250])
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=10., azim=45)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

def plot_intermediate(achieved_goals, desired_goals, r1, r2, r3, traj_point, save_path=None):
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = plt.axes(projection='3d')
    ag = np.array(achieved_goals) * 1000
    dg = np.array(desired_goals) * 1000
    ax.plot3D(r1[traj_point][:,0] * 1000, r1[traj_point][:,1] * 1000, r1[traj_point][:,2] * 1000, linewidth=2.0, c='blue')
    ax.plot3D(r2[traj_point][:,0] * 1000, r2[traj_point][:,1] * 1000, r2[traj_point][:,2] * 1000, linewidth=3.0, c='red')
    ax.plot3D(r3[traj_point][:,0] * 1000, r3[traj_point][:,1] * 1000, r3[traj_point][:,2] * 1000, linewidth=4.0, c='green')
    ax.plot3D(ag[:traj_point,0], ag[:traj_point,1], ag[:traj_point,2], marker='.', linestyle='-', label='achieved', c='magenta')
    ax.scatter(dg[:,0], dg[:,1], dg[:,2], marker='.', label='desired', c='black')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    #ax.set_xlim3d([0, 100])
    #ax.set_xticks([0, 50, 100])
    #ax.set_ylim3d([0, 100])
    #ax.set_yticks([0, 50, 100])
    #ax.set_zlim3d([0.0, 250])
    #ax.set_zticks([0, 50, 100, 150, 200, 250])
    ax.set_box_aspect([1,1,1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')

def plot_path_only(achieved_goals, desired_goals, save_path=None):
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = plt.axes(projection='3d')
    ag = np.array(achieved_goals) * 1000
    dg = np.array(desired_goals) * 1000
    ax.plot3D(ag[:,0], ag[:,1], ag[:,2], marker='.', linestyle='-', label='achieved', c='magenta')
    ax.scatter(dg[:,0], dg[:,1], dg[:,2], marker='.', label='desired', c='black')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    x_max = np.max(np.concatenate((ag[:, 0], dg[:, 0])))
    y_max = np.max(np.concatenate((ag[:, 1], dg[:, 1])))
    z_max = np.max(np.concatenate((ag[:, 2], dg[:, 2])))
    x_min = np.min(np.concatenate((ag[:, 0], dg[:, 0])))
    y_min = np.min(np.concatenate((ag[:, 1], dg[:, 1])))
    z_min = np.min(np.concatenate((ag[:, 2], dg[:, 2])))
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0

    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1,1,1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    ax.set_box_aspect([1,1,1])

def compare_paths_only(ag_1, ag_2, dg, save_path=None):
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = plt.axes(projection='3d')
    ag_1 = np.array(ag_1) * 1000
    dg = np.array(dg) * 1000
    ag_2 = np.array(ag_2) * 1000
    ax.plot3D(ag_1[:,0], ag_1[:,1], ag_1[:,2], marker='.', linestyle='-', label='Jacobian', c='blue')
    ax.plot3D(ag_2[:,0], ag_2[:,1], ag_2[:,2], marker='.', linestyle='-', label='DeepRL', c='purple')
    ax.scatter(dg[:,0], dg[:,1], dg[:,2], marker='.', label='Desired', c='black')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    #ax.legend()
    x_max = np.max(np.concatenate((ag_1[:, 0], ag_2[:, 0], dg[:, 0])))
    y_max = np.max(np.concatenate((ag_1[:, 1], ag_2[:, 1], dg[:, 1])))
    z_max = np.max(np.concatenate((ag_1[:, 2], ag_2[:, 2], dg[:, 2])))
    x_min = np.min(np.concatenate((ag_1[:, 0], ag_2[:, 0], dg[:, 0])))
    y_min = np.min(np.concatenate((ag_1[:, 1], ag_2[:, 1], dg[:, 1])))
    z_min = np.min(np.concatenate((ag_1[:, 2], ag_2[:, 2], dg[:, 2])))
    max_range = np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() / 2.0
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1,1,1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    ax.set_box_aspect([1,1,1])


# Animation functions
def update_animation(num, ax, tube1, tube2, tube3, ag_p, achieved_goals, desired_goals, training_step, r1, r2, r3, title=True):
    if num >= (len(achieved_goals)-1):
        # To run extra frames at the end
        num = len(achieved_goals)-1

    tube1.set_data(r1[num][:,0] * 1000, r1[num][:,1] * 1000)
    tube1.set_3d_properties(r1[num][:,2] * 1000)

    tube2.set_data(r2[num][:,0] * 1000, r2[num][:,1] * 1000)
    tube2.set_3d_properties(r2[num][:,2] * 1000)

    tube3.set_data(r3[num][:,0] * 1000, r3[num][:,1] * 1000)
    tube3.set_3d_properties(r3[num][:,2] * 1000)

    ag_p.set_data(achieved_goals[:num+1,0], achieved_goals[:num+1,1])
    ag_p.set_3d_properties(achieved_goals[:num+1,2])

    error = np.linalg.norm(achieved_goals[num, :] - desired_goals[num, :])
    if title:
        ax.set_title('Training: ' + str(int(training_step / 3e6 * 100)) + '%\nTolerance: ' +
                     str(round(decay_goal_tolerance(training_step) * 1000, 2)) + ' mm\nError: ' +
                     str(round(error, 2)) + ' mm',
                     fontsize=18, color='black', y=0.9)
    ax.view_init(elev=10., azim=45 + num)
    ax.set_xlim3d([-100, 100])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_ylim3d([-100, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_zlim3d([0, 250])
    ax.set_zticks([0, 50, 100, 150, 200, 250])
    ax.set_box_aspect([1,1,1])
    return [tube1, tube2, tube3, ag_p]

def animate_trajectory(achieved_goals, desired_goals, r1, r2 ,r3, training_step=None, tol=False, title=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ag = np.array(achieved_goals) * 1000
    dg = np.array(desired_goals) * 1000
    tube1, = ax.plot3D(r1[0][:, 0] * 1000, r1[0][:, 1] * 1000, r1[0][:, 2] * 1000, linewidth=2.0, c='#2596BE')
    tube2, = ax.plot3D(r2[0][:, 0] * 1000, r2[0][:, 1] * 1000, r2[0][:, 2] * 1000, linewidth=3.0, c='#D62728')
    tube3, = ax.plot3D(r3[0][:, 0] * 1000, r3[0][:, 1] * 1000, r3[0][:, 2] * 1000, linewidth=4.0, c='#2CA02C')
    if tol:
        ax.scatter(dg[0,0], dg[0,1], dg[0,2], label='desired', s=decay_goal_tolerance(training_step) * 5e4, marker='o',
                   facecolor=(0,0,0,0), edgecolors='black')
    ax.plot3D(dg[:,0], dg[:,1], dg[:,2], marker='.', linestyle=':', label='desired', c='magenta')
    ag_p, = ax.plot3D(ag[0,0], ag[0,1], ag[0,2], marker='.', linestyle=':', label='achieved', c='black')

    ani = animation.FuncAnimation(fig, update_animation, (len(achieved_goals)-1) + 10, fargs=[ax, tube1, tube2, tube3,
                                                                                     ag_p, ag, dg, training_step, r1, r2, r3, title])
    return ani

def generic_update_animation(num, ax, tube1, tube2, tube3, system, ag_p, achieved_goals, desired_goals, training_step, r1, r2, r3, title=True):
    if num >= (len(achieved_goals)-1):
        # To run extra frames at the end
        num = len(achieved_goals)-1

    tube1.set_data(r1[num][:,0] * 1000, r1[num][:,1] * 1000)
    tube1.set_3d_properties(r1[num][:,2] * 1000)

    tube2.set_data(r2[num][:,0] * 1000, r2[num][:,1] * 1000)
    tube2.set_3d_properties(r2[num][:,2] * 1000)

    tube3.set_data(r3[num][:,0] * 1000, r3[num][:,1] * 1000)
    tube3.set_3d_properties(r3[num][:,2] * 1000)

    ag_p.set_data(achieved_goals[:num+1,0], achieved_goals[:num+1,1])
    ag_p.set_3d_properties(achieved_goals[:num+1,2])

    error = np.linalg.norm(achieved_goals[num, :] - desired_goals[num, :])
    if title:
        ax.set_title('Training: ' + str(int(training_step / 3e6 * 100)) + '%\nTolerance: ' +
                     str(round(decay_goal_tolerance(training_step) * 1000, 2)) + ' mm\nError: ' +
                     str(round(error, 2)) + ' mm',
                     fontsize=18, color='black', y=0.9)
    ax.view_init(elev=10., azim=45 + num)
    ax.set_xlim3d([-100, 100])
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_ylim3d([-100, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_zlim3d([0, 250])
    ax.set_zticks([0, 50, 100, 150, 200, 250])
    ax.set_box_aspect([1,1,1])
    return [tube1, tube2, tube3, ag_p]

def generic_animate_trajectory(achieved_goals, desired_goals, r1, r2, r3, system, training_step=None, tol=False, title=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ag = np.array(achieved_goals) * 1000
    dg = np.array(desired_goals) * 1000
    tube1, = ax.plot3D(r1[0][:, 0] * 1000, r1[0][:, 1] * 1000, r1[0][:, 2] * 1000, linewidth=2.0, c=plt.cm.tab20c(system*4))
    tube2, = ax.plot3D(r2[0][:, 0] * 1000, r2[0][:, 1] * 1000, r2[0][:, 2] * 1000, linewidth=3.0, c=plt.cm.tab20c(system*4 + 1))
    tube3, = ax.plot3D(r3[0][:, 0] * 1000, r3[0][:, 1] * 1000, r3[0][:, 2] * 1000, linewidth=4.0, c=plt.cm.tab20c(system*4 + 2))
    if tol:
        ax.scatter(dg[0,0], dg[0,1], dg[0,2], label='desired', s=decay_goal_tolerance(training_step) * 5e4, marker='o',
                   facecolor=(0,0,0,0), edgecolors='black')
    ax.plot3D(dg[:,0], dg[:,1], dg[:,2], marker='.', linestyle=':', label='desired', c='magenta')
    ag_p, = ax.plot3D(ag[0,0], ag[0,1], ag[0,2], marker='.', linestyle=':', label='achieved', c='black')

    ani = animation.FuncAnimation(fig, generic_update_animation, (len(achieved_goals)-1) + 10, fargs=[ax, tube1, tube2, tube3, system,
                                                                                              ag_p, ag, dg, training_step, r1, r2, r3, title])
    return ani
