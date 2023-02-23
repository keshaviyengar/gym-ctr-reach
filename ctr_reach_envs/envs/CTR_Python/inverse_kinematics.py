import numpy as np
from ctr_reach_envs.envs.CTR_Python.Tube import Tube
from ctr_reach_envs.envs.CTR_Python.CTR_Model import CTR_Model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
import time
from ctr_reach_envs.src.plotting_utils import animate_trajectory, plot_trajectory, plot_path_only, plot_intermediate
from ctr_reach_envs.src.paths import velocity_based_line_traj

from ctr_reach_envs.envs.model import Model


class JacobianIk(object):
    def __init__(self, system_parameters, K_p, damping_constant, damping, home_offset, max_retraction):
        # Defining parameters of each tube, numbering starts with the most inner tube
        # length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature, y_curvature
        self.system_parameters = system_parameters
        self.tube1 = system_parameters[0][0]
        self.tube2 = system_parameters[0][1]
        self.tube3 = system_parameters[0][2]
        self.K_p = K_p
        self.damping_constant = damping_constant
        self.damping = damping
        self.eps = 1e-6
        self.home_offset = home_offset
        self.max_retraction = max_retraction

    def extension_limits(self, betas):
        # Apply extension joint constraints, rotation constraints applied through joint_spaces.
        tube_lengths = np.array([self.tube1.L, self.tube2.L, self.tube3.L])
        max_length = -self.home_offset - self.max_retraction
        # Ensure all joints below zero
        betas[betas > 0] = 0.0
        # Ensure each joint is not below the maximum length
        betas[0] = np.clip(betas[0], max_length[0], -self.home_offset[0])
        betas[1] = np.clip(betas[1], max_length[1], -self.home_offset[1])
        betas[2] = np.clip(betas[2], max_length[2], -self.home_offset[2])
        for i in range(1, 3):
            # Ordering is reversed, since we have innermost as last whereas in constraints its first.
            # Bi-1 <= Bi
            # Bi-1 >= Bi - Li-1 + Li
            betas[i - 1] = min(betas[i - 1], betas[i] + self.eps)
            betas[i - 1] = max(betas[i - 1], tube_lengths[i] - tube_lengths[i - 1] + betas[i] + self.eps)
        return betas

    def ik_solver(self, x_d, q):
        error_threshold = 5.0e-4
        max_steps = 10
        CTR = Model(self.system_parameters)
        x_c = CTR.forward_kinematics(q, 0)
        x_d_array = x_d
        x_c_array = x_c
        q_array = q
        for i in range(0,max_steps):
            print(q)
            del_x_d = x_d - x_c
            J = CTR.jac(q, 0)
            x_c = CTR.forward_kinematics(q, 0)
            K_p = self.K_p * np.eye(3)
            if self.damping:
                inv_J = np.linalg.pinv(np.transpose(J) @ J + self.damping_constant * np.eye(3)) @ np.transpose(J)
            else:
                inv_J = np.linalg.pinv(J)
            if not np.isfinite(np.linalg.cond(J)):
                print("J inverse is ill-conditioned.")
            del_q = (inv_J @ (del_x_d.reshape(3,1) + K_p @ (x_d - x_c).reshape(3, 1))).flatten()
            q += del_q
            #q[:3] = self.extension_limits(q[:3] + del_q[:3])
            if np.any(q[:3] > 0.0):
                print(q[:3])
                print('Joint limits reached...')
                break
            print('error: ' + str(np.linalg.norm(x_d - x_c)))
            x_d_array = np.append(x_d_array, x_d, axis=0)
            x_c_array = np.append(x_c_array, x_c, axis=0)
            q_array = np.append(q_array, q, axis=0)
            if np.linalg.norm(x_d - x_c) < error_threshold:
                break
        return x_d_array.reshape(-1, 3), x_c_array.reshape(-1, 3), q_array.reshape(-1, 6)

    def path_following(self, path_array, q):
        CTR = Model(self.system_parameters)
        x_c = CTR.forward_kinematics(q, 0)
        del_x_d_array = np.diff(path_array, axis=0, prepend=x_c.reshape(1,3))
        x_d_array = path_array[0,:].reshape(1,3)
        x_c_array = x_c
        q_array = q.reshape(1,6)
        for i in range(0, path_array.shape[0]):
            J = CTR.jac(q, 0)
            x_c = CTR.forward_kinematics(q, 0)
            x_d = path_array[i,:]
            K_p = self.K_p * np.eye(3)
            del_x_d = del_x_d_array[i,:].reshape(3,1)
            if self.damping:
                inv_J = np.transpose(J) @ np.linalg.pinv(J@ np.transpose(J) + self.damping_constant * np.eye(3))
            else:
                inv_J = np.linalg.pinv(J)
            if not np.isfinite(np.linalg.cond(J)):
                print("J inverse is ill-conditioned.")
            del_q = (inv_J @ (del_x_d.reshape(3,1) + K_p @ (x_d - x_c).reshape(3, 1))).flatten()
            q += del_q
            #q[:3] = self.extension_limits(q[:3] + del_q[:3])
            #q[3:] = q[3:]
            if np.any(q[:3] > 0.0):
                print(q[:3])
                print('Joint limits reached...')
                break
            print('error: ' + str(np.linalg.norm(x_d - x_c)))
            x_d_array = np.concatenate((x_d_array, x_d.reshape(1, 3)), axis=0)
            q_array = np.concatenate((q_array, q.reshape(1, 6)), axis=0)
            x_c_array = np.append(x_c_array, x_c, axis=0)
        return x_d_array.reshape(-1, 3), x_c_array.reshape(-1, 3), q_array.reshape(-1, 6)



if __name__ == '__main__':
    solve_ik = False
    solve_path_following = True
    # Defining parameters of each tube, numbering starts with the most inner tube
    # length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature, y_curvature
    #tube1 = Tube(431e-3, 103e-3, 2 * 0.35e-3, 2 * 0.55e-3, 6.43e+10, 2.50e+10, 21.3, 0)
    #tube2 = Tube(332e-3, 113e-3, 2 * 0.7e-3, 2 * 0.9e-3, 5.25e+10, 2.14e+10, 13.108, 0)
    #tube3 = Tube(174e-3, 134e-3, 2e-3, 2 * 1.1e-3, 4.71e+10, 2.97e+10, 3.5, 0)

    tube1 = Tube(431e-3, 103e-3, 2 * 0.35e-3, 2 * 0.55e-3, 10.25e+10, 18.79e+10, 21.3, 0)
    tube2 = Tube(332e-3, 113e-3, 2 * 0.7e-3, 2 * 0.9e-3, 68.6e+10, 11.53e+10, 13.1, 0)
    tube3 = Tube(174e-3, 134e-3, 2e-3, 2 * 1.1e-3, 16.96e+10, 14.25e+10, 3.5, 0)
    # initial twist (for ivp solver)
    uz_0 = np.array([0.0, 0.0, 0.0])
    u1_xy_0 = np.array([[0.0], [0.0]])
    f = np.array([0, 0, 0]).reshape(3, 1)
    # Control parameters
    K_p = 2
    damping_constant = 0
    damping = False
    if solve_ik:
        q = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
        # Initial position of joints
        q_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        del_q = np.array([-0.010, -0.005, 0.0, 0, 0, 0])
        CTR = CTR_Model(tube1, tube2, tube3, f, q + del_q, q_0, 0.01, 1)
        J = CTR.jac(np.concatenate((u1_xy_0, uz_0), axis=None))
        x_d = CTR.r[-1, :]
        # Single desired goal IK
        jacobian_ik = JacobianIk(tube1, tube2, tube3, K_p, damping_constant, damping)
        x_d_array, x_c_array, q_array = jacobian_ik.ik_solver(x_d, q, q_0, uz_0, u1_xy_0)
        # Plot single IK solutions
        #plot_traj(x_d_array, x_c_array)

    if solve_path_following:
        q = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
        # Initial position of joints
        q_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        home_offset = np.zeros(3)
        max_retraction = np.array([-tube1.L, -tube2.L, -tube3.L])
        jacobian_ik = JacobianIk(tube1, tube2, tube3, K_p, damping_constant, damping, home_offset, max_retraction)
        CTR = CTR_Model(tube1, tube2, tube3, f, q, q_0, 0.01, 1)
        J = CTR.jac(np.concatenate((u1_xy_0, uz_0), axis=None))
        x_d = CTR.r[-1, :]
        v = [0.001, -0.001, 0]
        path_x, path_y, path_z = velocity_based_line_traj(5, 10, v, x_d[0], x_d[1], x_d[2])
        path_array = np.vstack((path_x, path_y, path_z)).T
        x_d_array, x_c_array, q_array = jacobian_ik.path_following(path_array, q, q_0, uz_0, u1_xy_0)

        # Plot path following
        plot_path_only(x_c_array, x_d_array)
