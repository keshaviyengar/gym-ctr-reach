from copy import deepcopy

from ctr_reach_envs.envs.model_utils import *
from scipy.integrate import solve_ivp
from ctr_reach_envs.envs.CTR_Python import Segment

NUM_TUBES = 3


class Model(object):
    def __init__(self, system_parameters):
        self.system_parameters = deepcopy(system_parameters)
        self.current_sys_parameters = deepcopy(system_parameters)
        self.r = []
        self.r_transforms = []
        self.r1 = []
        self.r2 = []
        self.r3 = []

    def randomize_parameters(self, randomization):
        """
        Randomize tube parameters for each tube. Used for domain randomization.
        :param randomization: The percentage interval to sample from.
        """
        for i in range(len(self.current_sys_parameters)):
            self.current_sys_parameters[i][0] = sample_parameters(self.system_parameters[i][0], randomization)
            self.current_sys_parameters[i][1] = sample_parameters(self.system_parameters[i][1], randomization)
            self.current_sys_parameters[i][2] = sample_parameters(self.system_parameters[i][2], randomization)

    def forward_kinematics(self, joint, system, **kwargs):
        """
        q_0 = np.array([0, 0, 0, 0, 0, 0])
        # initial twist (for ivp solver)
        uz_0 = np.array([0.0, 0.0, 0.0])
        u1_xy_0 = np.array([[0.0], [0.0]])
        # force on robot tip along x, y, and z direction
        f = np.array([0, 0, 0]).reshape(3, 1)

        # Use this command if you wish to use initial value problem (ivp) solver (less accurate but faster)
        CTR = CTR_Model(self.systems[0][0], self.systems[0][1], self.systems[0][2], f, q, q_0, 0.01, 1)
        cost = CTR.minimize(np.concatenate((u1_xy_0, uz_0), axis=None))
        return CTR.r[-1]

        :param joint: Current joint position of robot [beta_0, ..., beta_2, alpha_0, ..., alpha_2]
        :param system: CTR system to use for forward kinematics.
        :param kwargs: Extra arguments.
        :return: End effector position or achieved goal of selected system.
        """
        # position of tubes' base from template (i.e., s=0)
        q_0 = np.array([0, 0, 0, 0, 0, 0])
        beta = joint[0:3] + q_0[0:3]

        segment = Segment(self.current_sys_parameters[system][0], self.current_sys_parameters[system][1],
                          self.current_sys_parameters[system][2], beta)

        r_0_ = np.array([0, 0, 0]).reshape(3, 1)
        alpha_1_0 = joint[3] + q_0[3]
        R_0_ = np.array(
            [[np.cos(alpha_1_0), -np.sin(alpha_1_0), 0], [np.sin(alpha_1_0), np.cos(alpha_1_0), 0], [0, 0, 1]]) \
            .reshape(9, 1)
        alpha_0_ = joint[3:].reshape(3, 1) + q_0[3:].reshape(3, 1)

        # initial twist
        uz_0_ = np.array([0, 0, 0])
        self.r, U_z, tip = self.ctr_model(system, uz_0_, alpha_0_, r_0_, R_0_, segment, beta)
        self.r1 = self.r[tip[1]:tip[0] + 1]
        self.r2 = self.r[tip[2]:tip[1] + 1]
        self.r3 = self.r[:tip[2] + 1]
        assert not np.any(np.isnan(self.r))
        return self.r[-1]

    def ode_eq(self, s, y, ux_0, uy_0, ei, gj):
        """
        Definition of ODE equation to solve overall backbone shape of CTR.
        :param s: Arc-length distance along backbone.
        :param y: y is 3 initial twist + 3 initial angle + 3 initial position + 9 initial rotation matrix
        :param ux_0: Initial curvature in x for segment.
        :param uy_0: Initial curvature in y for segment.
        :param ei: Youngs modulus times second moment of inertia
        :param gj: Shear modulus times polar moment of inertia
        :return: dydt set of differential equations
        """
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

    def ctr_model(self, system, uz_0, alpha_0, r_0, R_0, segmentation, beta):
        """
        :param system: Selected system to model.
        :param uz_0: Initial twist of backbone
        :param alpha_0: Initial angle of tubes
        :param r_0: Initial position of backbone
        :param R_0: Initial rotation matrix
        :param segmentation: Transition points where shape and internal moments continuity is enforced
        :param beta: Extension values for each tube.
        :return: r, u_z_end, tip_pos: Complete shape, twist at the tip and end-effector tip position.
        """
        tube1 = self.current_sys_parameters[system][0]
        tube2 = self.current_sys_parameters[system][1]
        tube3 = self.current_sys_parameters[system][2]
        Length = np.empty(0)
        r = np.empty((0, 3))
        u_z = np.empty((0, 3))
        alpha = np.empty((0, 3))
        span = np.append([0], segmentation.S)
        for seg in range(0, len(segmentation.S)):
            # Initial conditions, 3 initial twist + 3 initial angle + 3 initial position + 9 initial rotation matrix
            y_0 = np.vstack((uz_0.reshape(3, 1), alpha_0, r_0, R_0)).ravel()
            s_span = np.linspace(span[seg], span[seg + 1] - 1e-6, num=30)
            #s = odeint(self.ode_eq, y_0, s_span, args=(
            #    segmentation.U_x[:, seg], segmentation.U_y[:, seg], segmentation.EI[:, seg], segmentation.GJ[:, seg]),
            #           tfirst=True)
            if np.all(np.diff(s_span) < 0):
                print("s_span not sorted correctly. Resorting...")
                print("linespace: ", s_span[seg], s_span[seg+1] - 1e-6)
                s_span = np.sort(s_span)
            sol = solve_ivp(fun=lambda s, y: self.ode_eq(s, y, segmentation.U_x[:, seg], segmentation.U_y[:, seg],
                                                         segmentation.EI[:, seg], segmentation.GJ[:, seg]),
                            t_span=(min(s_span), max(s_span)), y0=y_0, t_eval=s_span)
            if sol.status == -1:
                print(sol.message)
            s = np.transpose(sol.y)
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

    # Estimating Jacobian with respect to inputs (tubes' rotation and translation)
    def jac(self, q, system):
        eps = 1.e-4
        jac = np.zeros((3, 6))
        r = self.forward_kinematics(q, system)
        for i in range(0, 6):
            q[i] = q[i] + eps
            r_eps = self.forward_kinematics(q, system)
            r_perturb = (r_eps - r) / eps
            jac[:, i] = r_perturb.reshape(3, )
            q[i] = q[i] - eps
        return jac
