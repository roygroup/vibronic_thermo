""" vibronic Monte Carlo """

# system imports
import sys
import os
from os.path import join, abspath, dirname
import itertools as it
# import functools

# third party imports
import numpy as np
from numpy.random import default_rng
# from scipy import sparse
# from scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg import LinearOperator
# import matplotlib as mpl; mpl.use("pdf")  # needed for WSL2 (can also use Agg)
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri

# local imports

# ----------------------------- I/O Functions ---------------------------------
output_dir = abspath(join(dirname(__file__), 'output'))
os.makedirs(output_dir, exist_ok=True)  # make sure the output directory exists


def make_output_data_path(file_name):
    """ Factorize out construction of the default input data path.
    Allows for easier modification.
    """
    data_path = join(output_dir, 'mcmh')
    os.makedirs(data_path, exist_ok=True)  # make sure the output directory exists

    path = join(data_path, file_name)

    return path


def memory_address(x):
    """ Return the memory address of `x` as hex code. """
    return hex(id(x))

# --------------------------- Model Parameters --------------------------------

# constants
eV_per_K = 8.617333262e-5
kB = eV_per_K

# displaced model parameters (all in eV)
displaced = {
    'energy': [0.0996, 0.1996],
    'gamma': [0., 0.04, 0.08, 0.12, 0.16, 0.20],
    'lambda': 0.072,
    'w1': 0.02,
    'w2': 0.04,
}

jahn_teller = {
    'energy': [0.02999, 0.00333, 0.07666, 0.20999, 0.39667, 0.63135],
    'lambda': [0.00, 0.04, 0.08, 0.12, 0.16, 0.20],
    'w1': 0.03,
    'w2': 0.03,
}

# ------------------------------ Monte Carlo -----------------------------------


def build_b_matrix(basis_size):
    """ build the B matrix, a closed-path tri-diagonal matrix.
    The covariance matrix in terms of the B matrix as follows:
        cov(j) = 2 * S(j) - C(j) * B
    """
    shape = (basis_size, basis_size)
    Bmat = np.zeros(shape, float)

    for i, ip in it.product(range(basis_size), repeat=2):
        # upper diagonal
        if ip == (i+1):
            Bmat[i, ip] = 1.0

        # lower diagonal
        if ip == (i-1):
            Bmat[i, ip] = 1.0

    # fill in the corners
    Bmat[0, basis_size-1] = 1.0
    Bmat[basis_size-1, 0] = 1.0

    return Bmat


def calculate_o_matrix(parameters, q_args, system_args, stat_props):
    """ calculate g without trace i.e. O matrix """

    # unpack parameters
    q1, q2 = q_args

    # unpack parameters
    A = parameters['nof_surfaces']
    # N = parameters['nof_modes']
    P = parameters['nof_beads']
    # T = parameters['T']
    # beta = parameters['beta']
    tau = parameters['tau']

    # unpack parameters
    Ea_tilde = system_args['Ea_tilde']

    # displacement (N x A) versus (A x N)
    dja = system_args['flipped_displacement']
    # dja_samp = system_args['flipped_displacement_samp']

    F = stat_props['F']
    S = stat_props['S']
    C = stat_props['C']

    o_matrix = np.zeros((P, 2, 2), float)

    # unclear what this is for
    # Omat_E = np.zeros((P,2,2),float)

    for p, a in it.product(range(P), range(A)):

        x1 = q1[p] - dja[0, a]
        x1p = q1[(p+1) % P] - dja[0, a]

        x2 = q2[p] - dja[1, a]
        x2p = q2[(p+1) % P] - dja[1, a]

        o_matrix[p, a, a] = np.exp(-tau*Ea_tilde[a])

        # for i in range(N):
        o_matrix[p, a, a] *= np.exp(S[0]*(x1*x1p)-.5*C[0]*(x1**2+x1p**2))
        o_matrix[p, a, a] *= np.exp(S[1]*(x2*x2p)-.5*C[1]*(x2**2+x2p**2))

        # o_matrix[p, a, a] *= np.exp(S1*(x1*x1p)-.5*C1*(x1**2+x1p**2))
        # o_matrix[p, a, a] *= np.exp(S2*(x2*x2p)-.5*C2*(x2**2+x2p**2))

        # Omat_E[p,a,a]=Ea_tilde[a]+S1*(x1*x1p)*S1_prime-.5*C1*(x1**2+x1p**2)*C1_prime
        # Omat_E[p,a,a]+=S2*(x2*x2p)*S2_prime-.5*C2*(x2**2+x2p**2)*C2_prime

    # o_matrix = (F1*F2)*o_matrix
    o_matrix = np.prod(F)*o_matrix

    return o_matrix


def calculate_m_matrix(parameters, q2, model):
    """ x """

    # unpack parameters
    A = parameters['nof_surfaces']
    P = parameters['nof_beads']
    # T = parameters['T']
    # beta = parameters['beta']
    tau = parameters['tau']

    # M matrix
    m_matrix = np.zeros((P, A, A), float)

    # unclear what this is for
    # Mmat_E = np.zeros((P, A, A), float)

    Vmat = np.zeros((A, A), float)
    Vmat_diag = np.zeros((A, A), float)
    # Vmat_diag_E = np.zeros((A, A), float)

    # for the specific model
    if model == 'Displaced':
        coeffiecent = displaced['gamma'][system_index]

    if model == 'Jahn_Teller':
        coeffiecent = jahn_teller['lambda'][system_index]

    for p in range(P):

        Vmat[0, 1] = Vmat[1, 0] = coeffiecent * q2[p]

        Vval, Vvec = np.linalg.eigh(Vmat)

        for a in range(A):
            Vmat_diag[a, a] = np.exp(-tau*Vval[a])
            # Vmat_diag_E[a, a] = Vval[a]

        Vmatp = np.dot(Vvec, np.dot(Vmat_diag, np.transpose(Vvec)))
        # Vmatp_E = np.dot(Vvec,np.dot(Vmat_diag_E,np.transpose(Vvec)))

        for a, ap in it.product(range(A), repeat=2):
            m_matrix[p, a, ap] = Vmatp[a, ap]
            # Mmat_E[p, a, ap] = Vmatp_E[a, ap]

    return m_matrix


def prepare_models(parameters, model, system_index):
    """ x """
    # unpack parameters
    A, N = parameters['nof_surfaces'], parameters['nof_modes']

    frequencies = np.zeros(N)
    frequencies_samp = np.zeros_like(frequencies)

    displacement = np.zeros((A, N))
    displacement_samp = np.zeros_like(displacement)

    if model == 'Displaced':

        frequencies[0] = displaced['w1']
        frequencies[1] = displaced['w2']

        frequencies_samp[0] = frequencies[0]
        frequencies_samp[1] = frequencies[1]

        mode_1_index = 0

        # compute the system's displacement (d^a_j = -g^aa_j / w_j)
        # so since its +lambda*q1 for A=0 we flip the sign to -lambda*q1
        # and similarly for A=1
        displacement[0, mode_1_index] = -1.0 * displaced['lambda'] / frequencies[mode_1_index]
        displacement[1, mode_1_index] = +1.0 * displaced['lambda'] / frequencies[mode_1_index]

        # compute the rho/sampling displacement
        if system_index == 0:
            # special case
            displacement_samp[0, mode_1_index] = -1.0 * displaced['lambda']
            displacement_samp[1, mode_1_index] = +1.0 * displaced['lambda']

            displacement_samp[:] /= frequencies_samp[mode_1_index]

            # q2 contribution is zero for uncoupled system (system_index == 0)

        elif 0 < system_index < 6:
            # q1 contribution is zero for coupled systems (system_index >= 1)

            mode_2_index = 1
            displacement_samp[0, mode_2_index] = -1.0 * displaced['gamma'][system_index]
            displacement_samp[1, mode_2_index] = +1.0 * displaced['gamma'][system_index]

            displacement_samp[:] /= frequencies_samp[mode_2_index]

        else:
            raise Exception(f'Incorrect value for {system_index = }')

        # (E^aa)
        energy_raw = displaced['energy'].copy()

        # (Δ^a) shift in energy from completing the square eq 36
        energy_shift = np.zeros(A)
        energy_shift_samp = np.zeros_like(energy_shift)

        """
        for some reason PN only used the first mode in his original code??
        should ask him about this, bug or intentional?
        """

        # for i in range(N):
        #     energy_shift += -0.5 * frequencies[i] * (displacement[:, i]**2.0)

        energy_shift += -0.5 * frequencies[0] * (displacement[:, 0]**2.0)
        # energy_shift += -0.5 * frequencies[1] * (displacement[:, 1]**2.0)

        energy_shift_samp += -0.5 * frequencies_samp[0] * (displacement_samp[:, 0]**2.0)
        energy_shift_samp += -0.5 * frequencies_samp[1] * (displacement_samp[:, 1]**2.0)

        # the shifted energy (E^aa + Δ^a)
        Ea_tilde = energy_raw + energy_shift

        # set E_0 = 0
        Ea_tilde[:] -= Ea_tilde[0]

    if model == 'Jahn_Teller':

        frequencies[0] = jahn_teller['w1']
        frequencies[1] = jahn_teller['w2']

        frequencies_samp[0] = frequencies[0]
        frequencies_samp[1] = frequencies[1]

        mode_1_index = 0
        displacement[0, mode_1_index] = -1.0 * jahn_teller['lambda'][system_index] / frequencies[mode_1_index]
        displacement[1, mode_1_index] = +1.0 * jahn_teller['lambda'][system_index] / frequencies[mode_1_index]

        # (E^aa)
        energy_raw = np.full(shape=A, fill_value=jahn_teller['energy'][system_index])

        # (Δ^a) shift in energy from completing the square eq 36
        energy_shift = np.zeros(A)
        energy_shift_samp = np.zeros_like(energy_shift)

        energy_shift += -0.5 * frequencies[0] * (displacement[:, 0]**2.0)
        # energy_shift += -0.5 * frequencies[1] * (displacement[:, 1]**2.0)

        energy_shift_samp += -0.5 * frequencies_samp[0] * (displacement_samp[:, 0]**2.0)
        energy_shift_samp += -0.5 * frequencies_samp[1] * (displacement_samp[:, 1]**2.0)

        # the shifted energy (E^aa + Δ^a)
        Ea_tilde = energy_raw + energy_shift

        # set E_0 = 0
        # Ea_tilde[:] -= Ea_tilde[0]

    # flip?
    flipped_displacement = np.zeros((N, A), float)
    flipped_displacement_samp = np.zeros_like(flipped_displacement)

    for a, i in it.product(range(A), range(N)):
        flipped_displacement[i, a] = displacement[a, i]
        flipped_displacement_samp[i, a] = displacement_samp[a, i]

    if parameters['sampling_type'] == 'Uniform':
        Ea_tilde_samp = np.zeros_like(Ea_tilde)

    if parameters['sampling_type'] in ['GMD', 'GMD_reduced', 'Direct']:

        # the shifted energy (E^aa + Δ^a) for sampling
        Ea_tilde_samp = energy_raw + energy_shift_samp

        if model != 'Jahn_Teller':
            # set E_0 = 0
            Ea_tilde_samp[:] -= Ea_tilde_samp[0]

    args = {
        'Ea_tilde': Ea_tilde,
        'Ea_tilde_samp': Ea_tilde_samp,
        'frequencies': frequencies,
        'frequencies_samp': frequencies_samp,
        'flipped_displacement': flipped_displacement,
        'flipped_displacement_samp': flipped_displacement_samp,
    }

    return args


def prepare_probability_distributions(parameters, system_args):
    """ x """

    # unpack parameters
    A = parameters['nof_surfaces']
    beta = parameters['beta']
    tau = parameters['tau']

    Ea_tilde = system_args['Ea_tilde']
    Ea_tilde_samp = system_args['Ea_tilde_samp']

    # initialize
    Prob_e = np.zeros(A)
    Prob_a = np.zeros(A)
    Prob_a_all = np.zeros(A)

    if parameters['sampling_type'] == 'Uniform':
        Prob_e[:] = np.exp(-tau*Ea_tilde[:])
        Prob_e /= np.sum(Prob_e)  # normalize

    if parameters['sampling_type'] in ['GMD', 'GMD_reduced', 'Direct']:
        Prob_a = np.exp(-beta*Ea_tilde_samp)
        Prob_a /= np.sum(Prob_a)  # normalize

    if parameters['sampling_type'] == 'Direct':
        Prob_a_all = np.exp(-tau*Ea_tilde_samp)
        Prob_a_all /= np.sum(Prob_a_all)  # normalize

    d = {
        'Prob_e': Prob_e,
        'Prob_a': Prob_a,
        'Prob_a_all': Prob_a_all,
    }

    return d


def prepare_statistical_constants(parameters, cur_state, system_args):
    """ Calculate hyperbolic prefactors, means and variances. """

    freq = system_args['frequencies']
    freq_samp = system_args['frequencies_samp']

    Ea_tilde_samp = system_args['Ea_tilde_samp']

    # unpack parameters
    # A = parameters['nof_surfaces']
    N = parameters['nof_modes']

    P = parameters['nof_beads']
    # T = parameters['T']
    beta = parameters['beta']
    tau = parameters['tau']

    # d/d tau

    C = np.zeros(N)
    C[:] = 1.0 / np.tanh(tau*freq[:])

    C_prime = np.zeros(N)
    C_prime[:] = -1.0 * (freq[:] / np.sinh(tau*freq[:])**2)

    S = np.zeros(N)
    S[:] = 1.0 / np.sinh(tau*freq[:])

    S_prime = np.zeros(N)
    S_prime[:] = -1.0 * (freq[:] * np.cosh(tau*freq[:]) / (np.sinh(tau*freq[:])**2))

    # -(a Cosh[a x] Csch[a x]^(3/2))/(2 Sqrt[2 Pi])

    F = np.zeros(N)
    F[:] = np.sqrt(S[:] / 2.0 / np.pi)

    # these don't get used?

    F_prime = np.zeros(N)
    F_prime[:] = freq[:] * np.cosh(tau*freq[:])
    F_prime *= (1.0 / np.sinh(tau*freq[:])**(3/2))
    F_prime /= (2.0 * np.sqrt(2.*np.pi))

    # F1_prime =- (freq[0]*np.cosh(tau*freq[0])*(1./np.sinh(tau*freq[0])**(3/2))/(2.*np.sqrt(2.*np.pi)))
    # F2_prime =- (freq[1]*np.cosh(tau*freq[1])*(1./np.sinh(tau*freq[1])**(3/2))/(2.*np.sqrt(2.*np.pi)))

    C_samp = np.zeros(N)
    C_samp[:] = 1. / np.tanh(tau*freq_samp[:])

    S_samp = np.zeros(N)
    S_samp[:] = 1. / np.sinh(tau*freq_samp[:])

    # these don't get used?
    # F_samp = np.zeros(N)
    # F_samp[:] = np.sqrt(S1_samp / 2.0 / np.pi)

    Bmat = build_b_matrix(P)

    mean_array = np.zeros((N, P))

    # build the covariance's as inverse
    inv_cov_q1 = np.zeros((P, P), float)
    inv_cov_q2 = np.zeros((P, P), float)

    for p in range(P):
        inv_cov_q1[p, p] = 2.0 * C_samp[0]
        inv_cov_q2[p, p] = 2.0 * C_samp[1]

        for pp in range(P):
            inv_cov_q1[p, pp] -= S_samp[0] * Bmat[p, pp]
            inv_cov_q2[p, pp] -= S_samp[1] * Bmat[p, pp]

    # invert covariance matrices
    cov_q1 = np.linalg.inv(inv_cov_q1)
    cov_q2 = np.linalg.inv(inv_cov_q2)

    # first step starts with all zeros
    x_q1 = np.zeros(parameters['nof_beads'], float)
    x_q2 = np.zeros(parameters['nof_beads'], float)

    x_list = [x_q1, x_q2]

    # exponential component of normal distributions
    xBx_q1 = np.dot(x_q1, np.dot(inv_cov_q1, x_q1))
    xBx_q2 = np.dot(x_q2, np.dot(inv_cov_q2, x_q2))

    wa_rhoa_old = 0.0
    wa_rhoa_old_all = 0.0

    if parameters['sampling_type'] in ['GMD', 'GMD_reduced']:

        pi_1 = np.exp(-0.5 * xBx_q1)
        pi_2 = np.exp(-0.5 * xBx_q2)

        # product of all the pi's (over the number of modes)
        wa_rhoa_old = pi_1*pi_2

        surface_index = cur_state['global_surface_index']

        # multiply by the exponential as per eq 47
        wa_rhoa_old *= np.exp(-beta*Ea_tilde_samp[surface_index])

    else:
        index = cur_state['per_bead_surface_index']

        wa_rhoa_old_all = np.exp(-0.5 * (xBx_q1 + xBx_q2))

        # modified above for mode bead sampling
        for p in range(P):
            wa_rhoa_old_all *= (-tau * Ea_tilde_samp[index[p]])

    d = {
        'F': F, 'S': S, 'C': C,
        'F_prime': F_prime, 'S_prime': S_prime, 'C_prime': C_prime,
        'mean_array': mean_array,
        'cov_q1': cov_q1, 'cov_q2': cov_q2,
        'inv_cov_q1': inv_cov_q1, 'inv_cov_q2': inv_cov_q2,
        'wa_rhoa_old': wa_rhoa_old,
        'wa_rhoa_old_all': wa_rhoa_old_all,
    }

    return d, x_list


def pick_new_state_dict():
    """ x """

    # this is how it would work when we are not doing the initialization
    x_q1 = np.random.multivariate_normal(mean_array[0, :], cov_q1)
    x_q2 = np.random.multivariate_normal(mean_array[1, :], cov_q2)

    scalar_surface_index = rng.choice(A, P, p=probs['Prob_a_all'])
    # gmd_surface_index = rng.choice(A, p=stat_props['Prob_a'])
    gmd_surface_index = 0

    return x


def compute_gmd_g_matrix(parameters, o_matrix, m_matrix):
    """ Compute g where Z = Tr[g(y)]. """

    g = np.identity(parameters['nof_surfaces'])

    for p in range(parameters['nof_beads']):
        g = np.dot(g, np.dot(o_matrix[p], m_matrix[p]))

    return g


def compute_single_g_scalar(parameters, a_index, o_matrix_new, m_matrix):
    """ Compute g where Z = Tr[g(y)]. """

    P = parameters['nof_beads']
    g_scalar = 1.0

    # compute the new scalar for the uniform/direct approach
    for p in range(P):

        m_index = (p, a_index[p], a_index[(p+1) % P])
        o_index = (p, a_index[p], a_index[p])

        g_scalar *= m_matrix[m_index] * o_matrix_new[o_index]

    return g_scalar


def q_from_x(x_list, system_args, probs):
    """ x """
    mode_1, mode_2 = 0, 1

    # displacement (N x A) versus (A x N)
    # dja = system_args['flipped_displacement']
    dja_samp = system_args['flipped_displacement_samp']

    # continuous variables
    surface_index = 0  # ground state
    q1 = x_list[mode_1] + dja_samp[mode_1, surface_index]
    q2 = x_list[mode_2] + dja_samp[mode_2, surface_index]

    ell = [
        q1,
        q2,
    ]

    return ell


def sampling_gmd(parameters, x_array, system_args, stat_props):
    """ x """

    #  ----- unpack -----
    A, P = parameters['nof_surfaces'], parameters['nof_beads']
    beta = parameters['beta']
    Ea_tilde_samp = system_args['Ea_tilde_samp']

    #  ----- do calculation -----

    pi_1 = np.exp(-.5*(np.dot(x_array[0], np.dot(stat_props['inv_cov_q1'], x_array[0]))))
    pi_2 = np.exp(-.5*(np.dot(x_array[1], np.dot(stat_props['inv_cov_q2'], x_array[1]))))

    wa_rhoa_new = np.exp(-beta*Ea_tilde_samp[gmd_surface_index])*pi_1*pi_2

    return wa_rhoa_new


def sampling_direct(parameters, q_args_new, system_args, stat_props, rng):
    """ x """

    #  ----- unpack -----
    A, P = parameters['nof_surfaces'], parameters['nof_beads']
    # N =  parameters['nof_modes']
    tau = parameters['tau']

    Ea_tilde_samp = system_args['Ea_tilde_samp']
    # displacement (N x A) versus (A x N)
    dja_samp = system_args['flipped_displacement_samp']

    mean1 = stat_props['mean_array'][0, :]
    mean2 = stat_props['mean_array'][1, :]

    #  ----- do calculation -----
    direct_surface_index = rng.choice(A, P, p=stat_props['Prob_a_all'])

    x1_new = np.random.multivariate_normal(mean1, stat_props['cov_q1'])
    x2_new = np.random.multivariate_normal(mean2, stat_props['cov_q2'])

    for p in range(P):
        # for j in range(N):
        q_args_new[0][p] = x1_new[p] + dja_samp[0, direct_surface_index[p]]
        q_args_new[1][p] = x2_new[p] + dja_samp[1, direct_surface_index[p]]

    x_cov_x_q1 = np.dot(x1_new, np.dot(stat_props['inv_cov_q1'], x1_new))
    x_cov_x_q2 = np.dot(x2_new, np.dot(stat_props['inv_cov_q2'], x2_new))

    wa_rhoa_new_all = np.exp(-0.5 * (x_cov_x_q1 + x_cov_x_q2))

    for p in range(P):
        wa_rhoa_new_all *= np.exp(-tau*Ea_tilde_samp[direct_surface_index[p]])

    return wa_rhoa_new_all, direct_surface_index


def _uniform_choose_new_mode():
    """ x """
    return


def _uniform_choose_new_surface():
    """ x """
    return


def sampling_uniform(parameters, q_args_new, q_args, system_args, cur_state, stat_props, rng):
    """ Uniform move proposal's
    This method moves a single bead of a single D.O.F.

    # S1*(x*xp)-.5*C1*(x**2+xp**2))*S1*(x*xm)-.5*C1*(x**2+xm**2))

    `q_args_new` is updated by-ref
    """

    #  ----- unpack -----
    A, N, P = parameters['nof_surfaces'], parameters['nof_modes'], parameters['nof_beads']

    # displacement (N x A) versus (A x N)
    dja = system_args['flipped_displacement']

    # statistical coefficients
    S, C = stat_props['S'], stat_props['C']

    # store something??
    # this seems to be useless since we overwrite the values immediately after?!?
    # for p in range(P):
    #     q1_new[p] = q1_old[p]
    #     q2_new[p] = q2_old[p]
    #     scalar_surface_index[p] = state_dict['scalar_a'][p]

    # choose from one of the modes or the electronic surfaces
    # if i == N that represents the electronic surfaces
    i = np.random.randint(0, N+1)

    # choose which bead to affect bead
    p = np.random.randint(0, P)

    # define neighbourhood beads
    x_old, x_plus, x_minus = 0.0, 0.0, 0.0

    #  ----- do calculation -----

    # current value
    old_surface = cur_state['global_surface_index']
    new_surface = cur_state['new_surface_index']
    new_a_per_p_index = cur_state['per_bead_surface_index'].copy()

    # mode 1
    if i == 0:
        # _uniform_choose_new_mode()

        x_plus = q_args[0][(p+1) % P]-dja[i, new_surface]
        x_minus = q_args[0][(p-1) % P]-dja[i, new_surface]

        mean = (0.5 * S[0] * (x_plus + x_minus)) / C[0]
        sigma = 1.0 / np.sqrt(2.0 * C[0])
        x_new = np.random.normal(mean,  sigma)

        q_args_new[0][p] = x_new + dja[i, new_surface]
        x_old = q_args[0][p] - dja[i, old_surface]

        pi_new = np.exp(-((x_new-mean)**2)/(2.*sigma**2))
        pi_old = np.exp(-((x_old-mean)**2)/(2.*sigma**2))

    # mode 2
    elif i == 1:
        x_plus = q_args[1][(p+1) % P]-dja[i, new_surface]
        x_minus = q_args[1][(p-1) % P]-dja[i, new_surface]

        mean = (0.5 * S[1] * (x_plus + x_minus)) / C[1]
        sigma = 1.0 / np.sqrt(2.0 * C[1])
        x_new = np.random.normal(mean,  sigma)

        q_args_new[1][p] = x_new + dja[i, new_surface]
        x_old = q_args[1][p] - dja[i, old_surface]

        pi_new = np.exp(-((x_new - mean)**2) / (2.0 * sigma**2))
        pi_old = np.exp(-((x_old - mean)**2) / (2.0 * sigma**2))

    # surfaces
    else:
        # _uniform_choose_new_surface()

        # proposed new surface
        new_surface = rng.choice(A, p=stat_props['Prob_e'])
        cur_state['new_surface_index'] = new_surface

        # update surface
        new_a_per_p_index[p] = new_surface
        pi_new = stat_props['Prob_e'][new_surface]
        pi_old = stat_props['Prob_e'][old_surface]

    return pi_new, pi_old, new_a_per_p_index


# --------------------------------- Main --------------------------------------


def main(model, system_index, mc_args, file_name):
    """ x """

    N_total = mc_args['N_total']
    N_equilibration = mc_args['N_equilibration']
    N_skip = mc_args['N_skip']
    sampling_type = mc_args['sampling_type']

    # basis sizes (store in dictionary for easy passing to functions)
    parameters = {
        'sampling_type': sampling_type,
        'nof_surfaces': 2,
        'nof_modes': 2,
        'nof_beads': 16,
        'T': 300.0,
    }

    parameters['beta'] = 1.0 / (kB*parameters['T'])
    parameters['tau'] = parameters['beta'] / float(parameters['nof_beads'])

    logfile = open(file_name, 'w')
    logfile.write(f"Model: {model}; System index = {system_index}\n")
    logfile.write(f"P = {parameters['nof_beads']}\n")
    logfile.write(f"tau (eV) = {parameters['tau']}\n")
    logfile.write(f"beta (eV) = {parameters['beta']}\n")
    # ----------------------------- x ---------------------------------

    # our random number generator
    rng = default_rng()

    system_args = prepare_models(parameters, model, system_index)

    prob_params = prepare_probability_distributions(parameters, system_args)

    logfile.write(f"Prob_e = {prob_params['Prob_e']}\n")
    logfile.write(f"Prob_a = {prob_params['Prob_a']}\n")
    logfile.write(f"Prob_a_all = {prob_params['Prob_a_all']}\n")

    # ----------------------------- x ---------------------------------
    # prepare the dictionary to store the Metropolis-Hastings values
    cur_state = {}

    if sampling_type in ['GMD', 'GMD_reduced']:
        cur_state['global_surface_index'] = 0

    if sampling_type in ['Direct', ]:
        cur_state['per_bead_surface_index'] = np.zeros(parameters['nof_beads'], int)

    if sampling_type in ['Uniform', ]:
        cur_state['global_surface_index'] = 0
        cur_state['new_surface_index'] = 0
        cur_state['per_bead_surface_index'] = np.zeros(parameters['nof_beads'], int)

    stat_props, x_list = prepare_statistical_constants(parameters, cur_state, system_args)

    # to cut down on argument passing (temporary)
    for key in ['Prob_e', 'Prob_a', 'Prob_a_all']:
        stat_props[key] = prob_params[key]

    # store in state dictionary
    cur_state['q'] = q_from_x(x_list, system_args, prob_params)

    o_matrix = calculate_o_matrix(parameters, cur_state['q'], system_args, stat_props)
    m_matrix = calculate_m_matrix(parameters, cur_state['q'][1], model)

    # ----------------------------- prepare initial parameters ---------------------------------
    A, N, P = parameters['nof_surfaces'], parameters['nof_modes'], parameters['nof_beads']

    number_of_accepted_states = 0

    # CALCULATE AN INITIAL g(x) value

    if sampling_type in ['GMD', 'GMD_reduced']:
        # compute the initial g value for the MCMH loop
        g_old = compute_gmd_g_matrix(parameters, o_matrix, m_matrix)
        cur_state['g_matrix_old'] = g_old
        cur_state['wa_rhoa_old'] = stat_props['wa_rhoa_old'].copy()

        cur_state['global_surface_index'] = 0

        # is this definition necessary?
        wa_rhoa_new = stat_props['wa_rhoa_old'].copy()

    # if we are sampling based on specific states and beads
    elif sampling_type in ['Direct', 'Uniform']:

        cur_state['per_bead_surface_index'] = np.zeros(parameters['nof_beads'], int)

        cur_state['g_scalar'] = compute_single_g_scalar(parameters, cur_state['per_bead_surface_index'], o_matrix, m_matrix)

        if sampling_type in ['Direct', ]:
            cur_state['wa_rhoa_old_all'] = stat_props['wa_rhoa_old_all'].copy()

            # is this definition necessary?
            wa_rhoa_new_all = stat_props['wa_rhoa_old_all'].copy()

        if sampling_type in ['Uniform', ]:
            cur_state['pi'] = 1.0

    # is this definition necessary?
    q_args_new = np.zeros_like(cur_state['q'])
    for i in range(N):
        q_args_new[i] = cur_state['q'][i].copy()

    # dq1, dq2 = 1.0, 1.0

    # gmd_surface_index = 0
    # uniform_index = 0
    # uniform_new_a = np.zeros_like(state_dict['per_bead_surface_index'])
    # direct_surface_index = np.zeros_like(state_dict['per_bead_surface_index'])

    # g_new_scalar = g_old_scalar
    # wa_rhoa_new = stat_props['wa_rhoa_old']
    # wa_rhoa_new_all = stat_props['wa_rhoa_old_all']

    # ----------------------------- Metropolis-Hastings loop ---------------------------------
    step_count = 0
    for step in range(N_total):

        if sampling_type in ['GMD', 'GMD_reduced']:
            mode_1, mode_2 = 0, 1

            # displacement (N x A) versus (A x N)
            dja_samp = system_args['flipped_displacement_samp']

            mean1 = stat_props['mean_array'][mode_1, :]
            mean2 = stat_props['mean_array'][mode_2, :]

            # propose a new state
            new_surface_index = rng.choice(A, p=stat_props['Prob_a'])
            x1 = np.random.multivariate_normal(mean1, stat_props['cov_q1'])
            x2 = np.random.multivariate_normal(mean2, stat_props['cov_q2'])
            x_list = [x1, x2]

            # this updates `q_args_new` in-place
            # shift to q co-ordinates
            # for j in range(N):
            for p in range(P):
                q_args_new[mode_1, p] = x_list[mode_1][p] + dja_samp[mode_1, new_surface_index]
                q_args_new[mode_2, p] = x_list[mode_2][p] + dja_samp[mode_2, new_surface_index]

            #  ----- do calculation -----

            pi_1 = np.exp(-0.5*(np.dot(x_list[mode_1], np.dot(stat_props['inv_cov_q1'], x_list[mode_1]))))
            pi_2 = np.exp(-0.5*(np.dot(x_list[mode_2], np.dot(stat_props['inv_cov_q2'], x_list[mode_2]))))

            wa_rhoa_new = np.exp(-parameters['beta']*system_args['Ea_tilde_samp'][new_surface_index])
            wa_rhoa_new *= pi_1 * pi_2
            # wa_rhoa_new = sampling_gmd(parameters, x_list, system_args, stat_props, new_surface_index)
            #  --------------------------

            # gmd approach
            o_matrix_new = calculate_o_matrix(parameters, q_args_new, system_args, stat_props)
            m_matrix = calculate_m_matrix(parameters, q_args_new[1], model)

            # compute the new g matrix for the GMD approach
            g_new = compute_gmd_g_matrix(parameters, o_matrix_new, m_matrix)

            # compute ratio
            if sampling_type == 'GMD':
                new_index = (new_surface_index, new_surface_index)
                old_index = (cur_state['global_surface_index'], cur_state['global_surface_index'])
                g_ratio = g_new[new_index] / cur_state['g_matrix_old'][old_index]
                sampling_ratio = cur_state['wa_rhoa_old'] / wa_rhoa_new
                ratio = g_ratio * sampling_ratio

            if sampling_type == 'GMD_reduced':
                ratio = np.trace(g_new) * stat_props['wa_rhoa_old']
                ratio /= np.trace(cur_state['g_matrix_old']) * wa_rhoa_new

        elif sampling_type == 'Direct':

            # reset new q's to old q's
            for i in range(N):
                q_args_new[i] = cur_state['q'][i].copy()

            # this updates `q_args_new` in-place
            wa_rhoa_new_all, direct_new_a = sampling_direct(parameters, q_args_new, system_args, stat_props, rng)

            o_matrix_new = calculate_o_matrix(parameters, q_args_new, system_args, stat_props)
            m_matrix = calculate_m_matrix(parameters, q_args_new[1], model)

            # compute the new g scalar
            g_new_scalar = compute_single_g_scalar(parameters, direct_new_a, o_matrix_new, m_matrix)

            # compute ratio
            g_ratio = g_new_scalar / cur_state['g_scalar']
            sampling_ratio = stat_props['wa_rhoa_old_all'] / wa_rhoa_new_all
            ratio = g_ratio * sampling_ratio

        elif sampling_type == 'Uniform':

            # reset new q's to old q's
            for i in range(N):
                q_args_new[i] = cur_state['q'][i].copy()

            # this updates `q_args_new` in-place
            pi_new, pi_old, new_a_per_p_index = sampling_uniform(
                parameters, q_args_new, cur_state['q'], system_args, cur_state, stat_props, rng
            )

            o_matrix_new = calculate_o_matrix(parameters, q_args_new, system_args, stat_props)
            m_matrix = calculate_m_matrix(parameters, q_args_new[1], model)

            # compute the new g scalar
            g_new_scalar = compute_single_g_scalar(parameters, new_a_per_p_index, o_matrix_new, m_matrix)

            # compute ratio
            ratio_num = g_new_scalar * pi_old
            ratio_denom = cur_state['g_scalar'] * pi_new
            ratio = ratio_num / ratio_denom

        # finally we can accept OR reject the proposed new state

        u = rng.random()  # our random number

        if (ratio >= u):

            number_of_accepted_states += 1  # record the accepted proposal

            mode_1, mode_2 = 0, 1
            # record the proposed state
            for p in range(P):
                cur_state['q'][mode_1][p] = q_args_new[mode_1][p]
                cur_state['q'][mode_2][p] = q_args_new[mode_2][p]

            if sampling_type in ['GMD', 'GMD_reduced']:

                cur_state['global_surface_index'] = new_surface_index
                cur_state['wa_rhoa_old'] = wa_rhoa_new

                for a, ap in it.product(range(A), repeat=2):
                    cur_state['g_matrix_old'][a, ap] = g_new[a, ap]

            if sampling_type in ['Direct', ]:
                cur_state['g_scalar'] = g_new_scalar
                cur_state['wa_rhoa_old_all'] = wa_rhoa_new_all

                for p in range(P):
                    cur_state['per_bead_surface_index'][p] = direct_new_a[p]

            if sampling_type in ['Uniform', ]:
                cur_state['g_scalar'] = g_new_scalar
                cur_state['pi'] = pi_new
                cur_state['global_surface_index'] = cur_state['new_surface_index']

                for p in range(P):
                    cur_state['per_bead_surface_index'][p] = new_a_per_p_index[p]

        # print to the file if necessary
        if step > N_equilibration and ((step % N_skip) == 0):

            mode_1, mode_2 = 0, 1
            q1 = cur_state['q'][mode_1]
            q2 = cur_state['q'][mode_2]

            for p in range(P):
                step_count += 1

                # string = ' '.join([str(step_count), str(q1[p]), str(q2[p]), str(index), str(a_old[p]), str(ratio)]) + '\n'

                if sampling_type in ['GMD', 'GMD_reduced']:
                    a_index = cur_state['global_surface_index']
                    string = f"{step_count} {q1[p]} {q2[p]} {a_index} {ratio}\n"

                if sampling_type in ['Direct', ]:
                    a_per_p_index = cur_state['per_bead_surface_index']
                    string = f"{step_count} {q1[p]} {q2[p]} {a_per_p_index[p]} {ratio}\n"

                if sampling_type in ['Uniform', ]:
                    a_index = cur_state['global_surface_index']
                    a_per_p_index = cur_state['per_bead_surface_index']
                    string = f"{step_count} {q1[p]} {q2[p]} {a_index} {a_per_p_index[p]} {ratio}\n"

                logfile.write(string)

    accept_ratio = number_of_accepted_states / N_total
    logfile.write(f"MC acceptance ratio = {accept_ratio}\n")
    logfile.close()


def profiling_code(model, system_index, mc_args):
    """ simple profiling """

    import cProfile
    import pstats

    filename = 'cProfile_output'
    cProfile.runctx(
        'main(model, system_index, mc_args)',
        globals(),
        locals(),
        filename
    )
    p = pstats.Stats(filename)
    p.strip_dirs().sort_stats("tottime").print_stats(8)
    p.strip_dirs().sort_stats("cumulative").print_stats(8)
    p.strip_dirs().sort_stats("cumulative").print_callees('calculate_distributions')
    # p.strip_dirs().sort_stats("cumulative").print_callers('calculate_distributions')

# ----------------------------- Plotting --------------------------------------


def write_spectrum_plotting_file(model, system_index, mc_args):
    """ x """

    N_total = mc_args['N_total']
    N_equilibration = mc_args['N_equilibration']
    N_skip = mc_args['N_skip']
    sampling_type = mc_args['sampling_type']

    # plotting command
    plotting_string = '\n'.join([
        # "set terminal png size 800,400",
        "set terminal png size 1200,800",
        f"set output './spectrum_{model:s}_{system_index:d}_{sampling_type:s}_{nof_points:d}_.png'",
        "set style data line",
        "set nologscale",
        "set xzeroaxis",
        # "set xr [ 0.2100000E+02: 0.100000E+02]",
        # "set yr [ -5: 80]",
        # "set xlabel 'Energy[eV]'",
        f"set title '{model:s}_{system_index:d} Spectrum, tau: {tau:d}.0 1, '",
        f"plot '{mctdh_file}.pl' using 1:3 lw 2 lc 'red' title 'MCTDH',\
         '{cc_file}.pl' every 6 using 1:3 with linespoints lc 'green' title 'CC',\
         '{sos_file}.pl' using 1:3 lc 'black' title 'SOS',\
          ",

        # f"plot '{mctdh_file}.pl' using 1:3 lw 4 lc 'red' title 'MCTDH',\
        #  '{x1}' every 6 using 1:3 with linespoints lc 'green' title 'CC halved',\
        #  '{x2}' every 4 using 1:3 with linespoints lc 'blue' title 'CC not halved',\
        #  '{x3}' every 6 using 1:3 lc 'black' title 'SOS halved',\
        #  '{x4}' every 4 using 1:3 lc 'orange' title 'SOS not halved',\
        # ",

        # f"plot '{mctdh_file}.pl' using 1:3 lw 4 lc 'red' title 'MCTDH',\
        #     '{x1}' every 6 using 1:3 with linespoints lc 'blue' title 'CC 12.5fs',\
        #     '{x2}' every 6 using 1:3 with linespoints lc 'magenta' title 'CC 25fs',\
        #     '{x3}' every 6 using 1:3 with linespoints lw 10 lc 'magenta' title 'CC 25fs (halved t)',\
        #     '{x4}' every 6 using 1:3 with linespoints lc 'green' title 'CC 50fs',\
        #     '{x5}' every 6 using 1:3 with linespoints lc 'orange' title 'CC 50fs (halved t)',\
        #     '{x6}' every 6 using 1:3 with linespoints lc 'yellow' title 'CC other',\
        # ",

        # f"plot '{mctdh_file}.pl' us 1:3 lc 'red' title 'MCTDH', '{cc_file}.pl' us 1:($3/6.582119564E-1) title 'CC' ",
        # f"plot '{cc_file}.pl' us 1:3 title 'CC' ",
    ])

    plotting_file = "spectrum_plotting.pl"
    # write the plotting commands to a file
    with open(plotting_file, 'w') as fp:
        fp.write(plotting_string)

    return plotting_file


# -----------------------------------------------------------------------------


def get_simple_user_input(model, system_index):
    """ x """

    nof_Args = len(sys.argv)

    if nof_Args > 1:
        # process the first argument
        # specify the lowercase character of the name of each model

        letter = str(sys.argv[1])
        if letter == 'd':
            model = 'Displaced'
        elif letter == 'j':
            model = 'Jahn_Teller'
        else:
            raise Exception(f"Only 'd' or 'j' are accepted, not {letter}")

        if nof_Args > 2:
            system_index = int(sys.argv[2])

    return model, system_index


if (__name__ == "__main__"):
    # default values
    model = ['Displaced', 'Jahn_Teller'][1]
    system_index = 5  # 0..5 for Displaced and Jahn-Teller

    # user input override (if any input)
    model, system_index = get_simple_user_input(model, system_index)
    assert 0 <= system_index <= 5, f'Currently only takes 0,1,2,3,4, or 5, not {system_index}'

    for s_type in ['Uniform', 'GMD_reduced', 'GMD', 'Direct']:
        mc_args = {
            'N_total': int(1e3),
            'N_equilibration': 10,
            'N_skip': int(1e1),
            'sampling_type': s_type,
        }

        # profiling_code(model, system_index, mc_args)

        file_name = f"{model}_{system_index}_{s_type}_MC.log"
        file_path = make_output_data_path(file_name)

        # run
        main(model, system_index, mc_args, file_path)

        # plot
        # write_spectrum_plotting_file(model, system_index, mc_args, file_path)
