import numpy as np

# Number of tubes is always 3
NUM_TUBES = 3


def single_joint2trig(joint):
    """
    Converting single tube extension and rotation to trigonometric representation.
    :param joint: Joint values of single tube [beta, alpha]
    :return: Trigonometric representation (cos(alpha), sin(alpha), beta)
    """
    return np.array([np.cos(joint[1]), np.sin(joint[1]), joint[0]])


def single_trig2joint(trig):
    """
    Converting single tube trigonometric representation to joint extension and rotation.
    :param trig: Input trigonometric representation as (cos(alpha), sin(alpha), beta)
    :return: return [beta, alpha]
    """
    return np.array([trig[2], np.arctan2(trig[1], trig[0])])


def rep2joint(rep):
    """
    Convert trigonometric representation of all tubes to simple joint representation.
    :param rep: Trigonometric representation of all tubes as array.
    :return: Simple joint representation [beta_0, ..., beta_2, alpha_0, ..., alpha_2]
    """
    rep = [rep[i:i + NUM_TUBES] for i in range(0, len(rep), NUM_TUBES)]
    beta = np.empty(NUM_TUBES)
    alpha = np.empty(NUM_TUBES)
    for tube in range(0, NUM_TUBES):
        joint = single_trig2joint(rep[tube])
        beta[tube] = joint[0]
        alpha[tube] = joint[1]
    return np.concatenate((beta, alpha))


def joint2rep(joint):
    """
    Convert simple joint representation to trigonometric representation.
    :param joint: Simple joints as [beta_0, ..., beta_2, alpha_0, ..., alpha_2]
    :return: Trigonoetric representation of all tubes [(cos(alpha_0), sin(alpha_0), beta_0), ... (cos(alpha_2), sin(alpha_2), beta_2a)]
    """
    rep = np.array([])
    betas = joint[:NUM_TUBES]
    alphas = joint[NUM_TUBES:]
    for beta, alpha in zip(betas, alphas):
        trig = single_joint2trig(np.array([beta, alpha]))
        rep = np.append(rep, trig)
    return rep


def ego2prop(joint):
    """
    Convert from egocentric joint representation to proprioceptive representation
    :param joint: Input joints are [beta_0, ..., beta_2, alpha_0, ..., alpha_2]_ego
    :return:[beta_0, ..., beta_2, alpha_0, ..., alpha_2]_prop joint representation
    """
    rel_beta = joint[:NUM_TUBES]
    rel_alpha = joint[NUM_TUBES:]
    betas = rel_beta.cumsum()
    alphas = rel_alpha.cumsum()
    return np.concatenate((betas, alphas))


def prop2ego(joint):
    """
    Convert from proprioceptive joint representation to egocentric representation
    :param joint: Input joints are [beta_0, ..., beta_2, alpha_0, ..., alpha_2]_prop
    :return:[beta_0, ..., beta_2, alpha_0, ..., alpha_2]_ego joint representation
    """
    betas = joint[:NUM_TUBES]
    alphas = joint[NUM_TUBES:]
    # Compute difference
    rel_beta = np.diff(betas, prepend=0)
    rel_alpha = np.diff(alphas, prepend=0)
    return np.concatenate((rel_beta, rel_alpha))

# Conversion between normalized and un-normalized joints
def B_to_B_U(B, *L_args):
    # Ensure number of tubes is either 2 or 3
    assert len(L_args) in [2, 3], 'length of length args: ' + str(len(L_args)) + str(L_args)
    num_tubes = len(L_args)
    B = np.append(B, 1)
    if num_tubes == 2:
        (L_1, L_2) = L_args
        M_B = np.array([[-L_1, 0],
                        [-L_1, L_1 - L_2]])

    else:
        (L_1, L_2, L_3) = L_args
        M_B = np.array([[-L_1, 0, 0],
                        [-L_1, L_1 - L_2, 0],
                        [-L_1, L_1 - L_2, L_2 - L_3]])

    normalized_B = np.block([[0.5 * M_B, 0.5 * np.matmul(M_B, np.ones((num_tubes, 1)))],
                             [np.zeros((1, num_tubes)), 1]])
    B_U = np.around(np.matmul(np.linalg.inv(normalized_B), B), 6)
    return B_U[:num_tubes]

# Conversion between normalized and un-normalized joints
def B_U_to_B(B_U, *L_args):
    # Ensure number of tubes is either 2 or 3
    assert len(L_args) in [2, 3], 'length of length args: ' + str(len(L_args)) + str(L_args)
    num_tubes = len(L_args)
    if num_tubes == 2:
        (L_1, L_2) = L_args
        B_U = np.append(B_U, 1)
        M_B = np.array([[-L_1, 0],
                        [-L_1, L_1 - L_2]])
    else:
        (L_1, L_2, L_3) = L_args
        B_U = np.append(B_U, 1)
        M_B = np.array([[-L_1, 0, 0],
                        [-L_1, L_1 - L_2, 0],
                        [-L_1, L_1 - L_2, L_2 - L_3]])

    normalized_B = np.block([[0.5 * M_B, 0.5 * np.matmul(M_B, np.ones((num_tubes, 1)))],
                             [np.zeros((1, num_tubes)), 1]])
    B = np.matmul(normalized_B, B_U)
    return B[:num_tubes]
