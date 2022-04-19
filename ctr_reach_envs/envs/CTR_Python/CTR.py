import numpy as np
from scipy.integrate import odeint
import time
from Tube import Tube
from Segment import Segment
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

start_time = time.time()

# Defining parameters of each tube, numbering starts with the most inner tube
# length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness,x_curvature, y_curvature
tube1 = Tube(431e-3, 103e-3, 2 * 0.35e-3, 2 * 0.55e-3, 6.4359738368e+10, 2.5091302912e+10, 21.3, 0)
tube2 = Tube(332e-3, 113e-3, 2 * 0.7e-3, 2 * 0.9e-3, 5.2548578304e+10, 2.1467424256e+10, 13.108, 0)
tube3 = Tube(174e-3, 134e-3, 2e-3, 2 * 1.1e-3, 4.7163091968e+10, 2.9788923392e+10, 3.5, 0)

# Joint variables
q = np.array([0.01, 0.015, 0.019, np.pi, np.pi * 5 / 2, np.pi / 2])
# Initial position of joints
q_0 = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
# position of tubes' base from template (i.e., s=0)
beta = q[0:3] + q_0[0:3]


# ode equation
def ode_eq(y, s, ux_0, uy_0, ei, gj):
    dydt = np.empty([18, 1])
    ux = np.empty([3, 1])
    uy = np.empty([3, 1])
    for i in range(0, 3):
        ux[i] = (1 / (ei[0] + ei[1] + ei[2])) * \
                (ei[0] * ux_0[0] * np.cos(y[3 + i] - y[3 + 0]) + ei[0] * uy_0[0] * np.sin(y[3 + i] - y[3 + 0]) +
                 ei[1] * ux_0[1] * np.cos(y[3 + i] - y[3 + 1]) + ei[1] * uy_0[1] * np.sin(y[3 + i] - y[3 + 1]) +
                 ei[2] * ux_0[2] * np.cos(y[3 + i] - y[3 + 2]) + ei[2] * uy_0[2] * np.sin(y[3 + i] - y[3 + 2]))
        uy[i] = (1 / (ei[0] + ei[1] + ei[2])) * \
                (-ei[0] * ux_0[0] * np.sin(y[3 + i] - y[3 + 0]) + ei[0] * uy_0[0] * np.cos(y[3 + i] - y[3 + 0]) +
                 -ei[1] * ux_0[1] * np.sin(y[3 + i] - y[3 + 1]) + ei[1] * uy_0[1] * np.cos(y[3 + i] - y[3 + 1]) +
                 -ei[2] * ux_0[2] * np.sin(y[3 + i] - y[3 + 2]) + ei[2] * uy_0[2] * np.cos(y[3 + i] - y[3 + 2]))

    for j in range(0, 3):
        if ei[j] == 0:
            dydt[j] = 0  # ui_z
            dydt[3 + j] = 0  # alpha_i
        else:
            dydt[j] = ((ei[j]) / (gj[j])) * (ux[j] * uy_0[j] - uy[j] * ux_0[j])  # ui_z
            dydt[3 + j] = y[j]  # alpha_i

    e3 = np.array([0, 0, 1]).reshape(3, 1)
    uz = y[0:3]
    R = np.array(y[9:]).reshape(3, 3)
    u_hat = np.array([(0, - uz[0], uy[0]), (uz[0], 0, -ux[0]), (-uy[0], ux[0], 0)])
    dr = np.dot(R, e3)
    dR = np.dot(R, u_hat).ravel()

    dydt[6] = dr[0]
    dydt[7] = dr[1]
    dydt[8] = dr[2]

    for k in range(3, 12):
        dydt[6 + k] = dR[k - 3]
    return dydt.ravel()


# CTR model
def ctr_model(uz_0, alpha_0, r_0, R_0, segmentation):
    Length = np.empty(0)
    r = np.empty((0, 3))
    u_z = np.empty((0, 3))
    alpha = np.empty((0, 3))
    span = np.append([0], segment.S)
    for seg in range(0, len(segmentation.S)):
        # Initial conditions, 3 initial twist + 3 initial angle + 3 initial position + 9 initial rotation matrix
        y_0 = np.vstack((uz_0.reshape(3, 1), alpha_0, r_0, R_0)).ravel()
        s_span = np.linspace(span[seg], span[seg + 1] - 1e-6, num=30)
        s = odeint(ode_eq, y_0, s_span, args=(segmentation.U_x[:, seg], segmentation.U_y[:, seg], segmentation.EI[:, seg], segmentation.GJ))
        Length = np.append(Length, s_span)
        u_z = np.vstack((u_z, s[:, (0, 1, 2)]))
        alpha = np.vstack((alpha, s[:, (3, 4, 5)]))
        r = np.vstack((r, s[:, (6, 7, 8)]))

        # new boundary conditions for next segment
        r_0 = r[-1, :].reshape(3, 1)
        R_0 = np.array(s[-1, 9:]).reshape(9, 1)
        uz_0 = u_z[-1, :].reshape(3, 1)
        alpha_0 = alpha[-1, :].reshape(3, 1)

    d_tip = np.array([tube1.L, tube2.L, tube3.L]) + beta
    u_z_end = np.array([0.0, 0.0, 0.0])
    tip_pos = np.array([0, 0, 0])
    for k in range(0, 3):
        b = np.argmax(Length >= d_tip[k] - 1e-3)  # Find where tube curve starts
        u_z_end[k] = u_z[b, k]
        tip_pos[k] = b

    return r, u_z_end, tip_pos


# initialize solved length, shape, curvatures, and twist angles
# segmenting the tubes
segment = Segment(tube1, tube2, tube3, beta)

r_0_ = np.array([0, 0, 0]).reshape(3, 1)
alpha_1_0 = q[3] + q_0[3]
R_0_ = np.array([[np.cos(alpha_1_0), -np.sin(alpha_1_0), 0], [np.sin(alpha_1_0), np.cos(alpha_1_0), 0], [0, 0, 1]])\
    .reshape(9, 1)
alpha_0_ = q[3:].reshape(3, 1) + q_0[3:].reshape(3, 1)

# initial twist
uz_0_ = np.array([0, 0, 0])

shape, U_z, tip = ctr_model(uz_0_, alpha_0_, r_0_, R_0_, segment)

print(tip)

# show execution time
print("--- %s seconds ---" % (time.time() - start_time))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(shape[:, 0], shape[:, 1], shape[:, 2], '-b', linewidth=2)
ax.plot(shape[:tip[1], 0], shape[:tip[1], 1], shape[:tip[1], 2], '-r', linewidth=3)
ax.plot(shape[:tip[2], 0], shape[:tip[2], 1], shape[:tip[2], 2], '-g', linewidth=4)
ax.auto_scale_xyz([min(shape[:, 0]), max(shape[:, 0])], [min(shape[:, 1]), max(shape[:, 0])],
                  [min(shape[:, 2]), max(shape[:, 2])])
plt.show()

# Save into csv file
#np.savetxt('/home/mohsen/git_ws/CTR_Control_Matlab/FileName.csv', shape,  delimiter=',')
