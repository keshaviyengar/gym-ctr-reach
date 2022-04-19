
import numpy as np
from Tube import Tube
from CTR_Model import CTR_Model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from matplotlib import cm
import matplotlib.pyplot as plt
import time

start_time = time.time()

# Defining parameters of each tube, numbering starts with the most inner tube
# length, length_curved, diameter_inner, diameter_outer, stiffness, torsional_stiffness, x_curvature, y_curvature
tube1 = Tube(431e-3, 103e-3, 2 * 0.35e-3, 2 * 0.55e-3, 6.43e+10, 2.50e+10, 21.3, 0)
tube2 = Tube(332e-3, 113e-3, 2 * 0.7e-3, 2 * 0.9e-3, 5.25e+10, 2.14e+10, 13.108, 0)
tube3 = Tube(174e-3, 134e-3, 2e-3, 2 * 1.1e-3, 4.71e+10, 2.97e+10, 3.5, 0)
# Joint variables
q = np.array([0.01, 0.015, 0.019, np.pi / 2, 5 * np.pi / 2, 3 * np.pi / 2])
# Initial position of joints
q_0 = np.array([-0.2858, -0.2025, -0.0945, 0, 0, 0])
# initial twist (for ivp solver)
uz_0 = np.array([0.0, 0.0, 0.0])
u1_xy_0 = np.array([[0.0], [0.0]])
# force on robot tip along x, y, and z direction
f = np.array([0, 0, 0]).reshape(3, 1)

# Use this command if you wish to use initial value problem (ivp) solver (less accurate but faster)
CTR = CTR_Model(tube1, tube2, tube3, f, q, q_0, 0.01, 1)
C = CTR.comp(np.concatenate((u1_xy_0, uz_0), axis=None))  # estimate compliance matrix
J = CTR.jac(np.concatenate((u1_xy_0, uz_0), axis=None))   # estimate jacobian matrix

# Use this command if you wish to use boundary value problem (bvp) solver (very accurate but slower)
u_init = CTR.minimize(np.concatenate((u1_xy_0, uz_0), axis=None))
C = CTR.comp(u_init)  # estimate compliance matrix
J = CTR.jac(u_init)   # estimate jacobian matrix

# plot the robot shape
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(CTR.r[:, 0], CTR.r[:, 1], CTR.r[:, 2], '-b',label='CTR Robot')
ax.auto_scale_xyz([np.amin(CTR.r[:, 0]), np.amax(CTR.r[:, 0]) + 0.01],
                  [np.amin(CTR.r[:, 1]), np.amax(CTR.r[:, 1]) + 0.01],
                  [np.amin(CTR.r[:, 2]), np.amax(CTR.r[:, 2]) + 0.01])
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
plt.grid(True)
plt.legend()
plt.show()
