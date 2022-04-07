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
    data_path = join(output_dir, 'data')
    os.makedirs(data_path, exist_ok=True)  # make sure the output directory exists

    path = join(data_path, file_name)

    return path

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


def calculate_o_matrix(parameters, q1, q2, system_args, stat_props):
    """ calculate g without trace i.e. O matrix """

    # unpack parameters
    A = parameters['nof_surfaces']
    # N = parameters['nof_modes']
    P = parameters['P']
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


def calculate_m_matrix(parameters, q2):
    """ x """

    # unpack parameters
    A = parameters['nof_surfaces']
    P = parameters['P']
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


def prepare_models(parameters, sampling_type, model, system_index):
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

    if model == 'Jahn_Teller':
        frequencies[0] = jahn_teller['w1']
        frequencies[1] = jahn_teller['w2']

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

    if sampling_type == 'Uniform':

        # energy for uniform sampling
        Ea_uniform = displaced['energy'].copy()

    # GMD or Direct
    else:
        # set E_0 = 0
        Ea_tilde[:] -= Ea_tilde[0]

        # flip?
        flipped_displacement = np.zeros((N, A), float)
        flipped_displacement_samp = np.zeros_like(flipped_displacement)

        for a, i in it.product(range(A), range(N)):
            flipped_displacement[i, a] = displacement[a, i]
            flipped_displacement_samp[i, a] = displacement_samp[a, i]

        # the shifted energy (E^aa + Δ^a) for sampling
        Ea_tilde_samp = energy_raw + energy_shift_samp

        # set E_0 = 0
        Ea_tilde_samp[:] -= Ea_tilde_samp[0]

        if sampling_type == 'GMD' or sampling_type == 'GMD_reduced':
            pass

        elif sampling_type == 'Direct':
            pass

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
    # P = parameters['P']
    # T = parameters['T']
    A = parameters['nof_surfaces']
    beta = parameters['beta']
    tau = parameters['tau']

    Ea_tilde = system_args['Ea_tilde']
    Ea_tilde_samp = system_args['Ea_tilde_samp']

    # D, gamma=0.08 Theta= 226.38994229156003  K model 2

    Prob_a = np.zeros(A)
    Prob_e = np.zeros(A)
    Prob_a_all = np.zeros(A)

    Prob_e[:] = np.exp(-tau*Ea_tilde[:])
    Prob_a[:] = np.exp(-beta*Ea_tilde_samp[:])
    Prob_a_all[:] = np.exp(-tau*Ea_tilde_samp[:])

    # Prob_a[a] = np.exp(-beta*(Ea_uniform[a]))

    # normalize the probabilities
    Prob_a /= np.sum(Prob_a)
    Prob_e /= np.sum(Prob_e)
    Prob_a_all /= np.sum(Prob_a_all)

    d = {
        'Prob_a': Prob_a,
        'Prob_a_all': Prob_a_all,
        'Prob_e': Prob_e,
    }

    return d


def prepare_statistical_constants(parameters, rng, system_args, probs):
    """ Calculate hyperbolic prefactors, means and variances. """

    freq = system_args['frequencies']
    freq_samp = system_args['frequencies_samp']

    Ea_tilde_samp = system_args['Ea_tilde_samp']

    # displacement (N x A) versus (A x N)
    # dja = system_args['flipped_displacement']
    dja_samp = system_args['flipped_displacement_samp']

    # unpack parameters
    A = parameters['nof_surfaces']
    N = parameters['nof_modes']

    P = parameters['P']
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

    # file path
    file_name = f"x12_{model}_{system_index}.dat"
    path = join(output_dir, file_name)

    # recommended numpy random number initialization
    # initial conditions
    # index=rng.choice(na,p=Prob_a)
    # try all zeros
    # sma initial condition for all methods

    x_q1_old = np.random.multivariate_normal(mean_array[0, :], cov_q1)
    x_q2_old = np.random.multivariate_normal(mean_array[1, :], cov_q2)

    surface_index = 0  # ground state
    a_old = rng.choice(A, P, p=probs['Prob_a_all'])

    for p in range(P):
        a_old[p] = 0
        x_q1_old[p] = 0.0
        x_q2_old[p] = 0.0

    # continuous variables
    q1 = x_q1_old + dja_samp[0, surface_index]
    q2 = x_q2_old + dja_samp[1, surface_index]

    # exponential component of normal distributions
    xBx_q1 = np.dot(x_q1_old, np.dot(inv_cov_q1, x_q1_old))
    xBx_q2 = np.dot(x_q2_old, np.dot(inv_cov_q2, x_q2_old))

    pi_1 = np.exp(-0.5 * xBx_q1)
    pi_2 = np.exp(-0.5 * xBx_q2)

    # product of all the pi's (over the number of modes)
    wa_rhoa_old = pi_1*pi_2

    # multiply by the exponential as per eq 47
    wa_rhoa_old *= np.exp(-beta*Ea_tilde_samp[surface_index])

    wa_rhoa_old_all = np.exp(-0.5 * (xBx_q1 + xBx_q2))

    # modified above for mode bead sampling
    for p in range(P):
        wa_rhoa_old_all *= (-tau * Ea_tilde_samp[a_old[p]])

    d = {
        'q1': q1,
        'q2': q2,
        'wa_rhoa_old': wa_rhoa_old,
        'wa_rhoa_old': wa_rhoa_old,
        'wa_rhoa_old_all': wa_rhoa_old_all,
    }

    return d


def main(model, system_index, N_total=10000, N_equilibration=100, N_skip=1, sampling_type='GMD'):

    # basis sizes (store in dictionary for easy passing to functions)
    parameters = {
        'nof_surfaces': 2,
        'nof_modes': 2,
        'P': 16,
        'T': 300.0,
    }
    parameters['beta'] = 1.0 / (kB*parameters['T'])
    parameters['tau'] = parameters['beta'] / float(parameters['P'])

    file_name = f"{model}_{system_index}_MC.log"
    logfile = open(file_name, 'w')
    logfile.write(f"Model: {model}; System index = {system_index}\n")
    logfile.write(f"P = {parameters['P']}\n")
    logfile.write(f"tau (eV) = {parameters['tau']}\n")
    logfile.write(f"beta (eV) = {parameters['beta']}\n")

    # our random number generator
    rng = default_rng()

    system_args = prepare_models(parameters, sampling_type, model, system_index)

    prob_param = prepare_probability_distributions(parameters, system_args)

    logfile.write(f"Prob_a = {prob_param['Prob_a']}\n")
    logfile.write(f"Prob_a_all = {prob_param['Prob_a_all']}\n")

    stat_props = prepare_statistical_constants(parameters, rng, system_args, prob_param)

    # ----------------------------- x ---------------------------------

    o_matrix = calculate_o_matrix(parameters, q1, q2, system_args, stat_props)
    m_matrix = calculate_m_matrix()

    # ----------------------------- Metropolis-Hastings loop ---------------------------------
    A, N, P = parameters['nof_surfaces'], parameters['nof_modes'], parameters['P']

    # compute the initial g value for the MCMH loop
    g_old = np.identity(A)
    for p in range(P):
        g_old = np.dot(g_old, np.dot(o_matrix[p], m_matrix[p]))

    # g calculate for state bead sampling
    g_old_scalar = 1.

    for p in range(P):

        m = m_matrix[p, a_old[p], a_old[(p+1) % data.beads]]

        eee1 = -tau*Ea_tilde[a_old[p]]
        sss1 = S1*(x1*x1p)-.5*C1*(x1**2+x1p**2)
        sss2 = S2*(x2*x2p)-.5*C2*(x2**2+x2p**2)
        exp1 = np.exp(eee1 + sss1 + sss2)

        g_old_scalar *= m * exp1

    accept = 0

    dq1, dq2 = 1.0, 1.0
    q1_new, q2_new = q1.copy(), q2.copy()
    pi_new, pi_old = 1.0, 1.0

    a_new = a_old.copy()
    g_new_scalar = g_old_scalar
    wa_rhoa_new = wa_rhoa_old
    wa_rhoa_new_all = wa_rhoa_old_all

    index_new = index

    step_count = 0
    for step in range(N_total):

        if Sampling_type == 'GMD' or Sampling_type == 'GMD_reduced':

            def sampling_gmd():
                """ x """
                index_new = rng.choice(na, p=Prob_a)
                x1_new = np.random.multivariate_normal(mean1, cov1)
                x2_new = np.random.multivariate_normal(mean2, cov2)

                for p in range(P):
                    q1_new[p] = x1_new[p]+dja_samp[0, index_new]
                    q2_new[p] = x2_new[p]+dja_samp[1, index_new]

                pi_1 = np.exp(-.5*(np.dot(x1_new, np.dot(cov1inv, x1_new))))
                pi_2 = np.exp(-.5*(np.dot(x2_new, np.dot(cov2inv, x2_new))))

                wa_rhoa_new = np.exp(-beta*Ea_tilde_samp[index_new])*pi_1*pi_2

                return wa_rhoa_new

        elif Sampling_type == 'Direct':

            def sampling_direct():
                """ x """

                x1_new = np.random.multivariate_normal(mean1, cov1)
                x2_new = np.random.multivariate_normal(mean2, cov2)
                a_new = rng.choice(na, P, p=Prob_a_all)

                for p in range(P):
                    q1_new[p] = x1_new[p]+dja_samp[0, a_new[p]]
                    q2_new[p] = x2_new[p]+dja_samp[1, a_new[p]]

                wa_rhoa_new_all = np.exp(-.5*(np.dot(x1_new, np.dot(cov1inv, x1_new)))-.5*(np.dot(x2_new, np.dot(cov2inv, x2_new))))

                for p in range(P):
                    wa_rhoa_new_all *= np.exp(-tau*Ea_tilde_samp[a_new[p]])

                return wa_rhoa_new

        elif Sampling_type == 'Uniform':

            def sampling_uniform():
                """ Uniform move proposal's
                This method moves a single bead of a single D.O.F.
                """

                # S1*(x*xp)-.5*C1*(x**2+xp**2))*S1*(x*xm)-.5*C1*(x**2+xm**2))

                # store something??
                for p in range(P):
                    q1_new[p] = q1[p]
                    q2_new[p] = q2[p]
                    a_new[p] = a_old[p]

                # choose from one of the modes or the electronic surfaces
                i = np.random.randint(0, nmodes+1)

                p = np.random.randint(0, P)  # choose a bead

                # define neighbourhood beads
                x_old, x_plus, x_minus = 0.0, 0.0, 0.0

                # mode 1
                if i == 0:
                    x_plus = q1[(p+1) % P]-dja[i, index_new]
                    x_minus = q1[(p-1) % P]-dja[i, index_new]

                    mean = (0.5*S1*(x_plus + x_minus))/C1
                    sigma = 1./np.sqrt(2.*C1)

                    x_new = np.random.normal(mean,  sigma)
                    q1_new[p] = x_new + dja[i, index_new]
                    x_old = q1[p] - dja[i, index]

                    pi_new = np.exp(-((x_new-mean)**2)/(2.*sigma**2))
                    pi_old = np.exp(-((x_old-mean)**2)/(2.*sigma**2))

                # mode 2
                elif i == 1:
                    x_plus = q2[(p+1) % P]-dja[i, index_new]
                    x_minus = q2[(p-1) % P]-dja[i, index_new]

                    mean = (0.5*S2*(x_plus + x_minus))/C2
                    sigma = 1./np.sqrt(2.*C2)

                    x_new = np.random.normal(mean,  sigma)
                    q2_new[p] = x_new + dja[i, index_new]
                    x_old = q2[p] - dja[i, index]

                    pi_new = np.exp(-((x_new-mean)**2)/(2.*sigma**2))
                    pi_old = np.exp(-((x_old-mean)**2)/(2.*sigma**2))

                # surfaces
                else:
                    # choose a surface
                    index_new = rng.choice(na, p=Prob_e)

                    # update surface
                    a_new[p] = index_new
                    pi_new = Prob_e[index_new]
                    pi_old = Prob_e[index]

                return

        o_matrix_new = calculate_o_matrix(q1_new, q2_new)
        m_matrix = calculate_m_matrix(q2_new)

        g_new = np.dot(o_matrix_new[0], m_matrix[0])
        for p in range(1, P):
            g_new = np.dot(g_new, np.dot(o_matrix_new[p], m_matrix[p]))

        g_new_scalar = 1.

        # compute the new scalar for the uniform approach
        for p in range(P):

            m_index = (p, a_new[p], a_new[(p+1) % P])
            o_index = (p, a_new[p], a_new[p])

            g_new_scalar *= m_matrix[m_index] * o_matrix_new[o_index]

        def compute_ratio():
            """ x """
            ratio = 1.

            if Sampling_type == 'GMD':
                ratio = g_new[index_new, index_new]/g_old[index, index]*wa_rhoa_old/wa_rhoa_new

            if Sampling_type == 'GMD_reduced':
                ratio = np.trace(g_new)*wa_rhoa_old/(np.trace(g_old)*wa_rhoa_new)

            if Sampling_type == 'Direct':
                ratio_num = g_new_scalar*wa_rhoa_old_all
                ratio_denom = g_old_scalar*wa_rhoa_new_all
                ratio = ratio_num/ratio_denom

            if Sampling_type == 'Uniform':
                ratio_num = g_new_scalar*pi_old
                ratio_denom = g_old_scalar*pi_new
                ratio = ratio_num/ratio_denom

            return ratio

        # finally we can accept OR reject the proposed new state

        u = rng.random()  # our random number

        if (ratio >= u):

            accept += 1  # record the accepted proposal

            # record the proposed state
            for p in range(P):
                q1[p] = q1_new[p]
                q2[p] = q2_new[p]
                a_old[p] = a_new[p]

            index = index_new

            for a, ap in it.product(range(na), repeat=2):
                g_old[a, ap] = g_new[a, ap]

            wa_rhoa_old = wa_rhoa_new
            g_old_scalar = g_new_scalar
            wa_rhoa_old_all = wa_rhoa_new_all
            pi_old = pi_new

        # print to the file if necessary
        if step > N_equilibration and ((step % N_skip) == 0):
            for p in range(P):
                step_count += 1

                # string = ' '.join([str(step_count), str(q1[p]), str(q2[p]), str(index), str(a_old[p]), str(ratio)]) + '\n'

                string = f"{step_count} {q1[p]} {q2[p]} {index} {a_old[p]} {ratio}\n"

                outx1x2.write(string)

    logfile.write('MC acceptance ratio = '+str(accept/N_total)+'\n')
    logfile.close()
# --------------------------------- Main --------------------------------------


# def main(model, system_index, mc_args):
#     """ x """

#     # mc_args = {
#     #     'N_total': int(4e5),
#     #     'N_equilibration': 10,
#     #     'N_skip': int(1e3),
#     #     'Sampling_type': ['Uniform', 'GMD_reduced', 'GMD', 'Direct'][0],
#     # }

#     # basis sizes (store in dictionary for easy passing to functions)
#     n1, n2, na = 10, 10, 2
#     basis = {'n1': n1, 'n2': n2, 'a': na}

#     # total size of product basis
#     N = n1*n2*na
#     basis['N'] = N

#     # modes only basis size
#     # n12 = n1*n2

#     h01, h02 = create_harmonic_matrices(basis)

#     # create Discrete Variable Representation grid
#     dvr_h_terms, q_mats, grids, T_list = create_dvr_grid(h01, h02, basis)

#     if fbr_flag:
#         # create Full Basis Representation h terms
#         fbr_h01 = np.diag(h01)
#         fbr_h02 = np.diag(h02)
#         fbr_h_terms = [fbr_h01, fbr_h02]
#     else:
#         fbr_h_terms = [None, None]

#     args = (
#         N, dvr_h_terms, fbr_h_terms, grids, q_mats,
#         system_index, model, basis, fbr_flag
#     )
#     if not fbr_flag:
#         dvr_H_total = build_full_hamiltonian(*args)
#     else:
#         dvr_H_total, fbr_H_total = build_full_hamiltonian(*args)

#     k_max = 100

#     assert k_max < N, (
#         f'The number of requested eigenvalues/vectors {k_max = } '
#         f'must be strictly < the basis size {N = }'
#     )

#     # diagonalize
#     # niter = 100
#     # evals, evecs = eigsh(A_total, k=k_max, which = 'SA', maxiter=niter)
#     evals, evecs = eigsh(dvr_H_total, k=k_max, which='SA')

#     n_short = 5
#     np.set_printoptions(precision=16)
#     print(f"First {n_short} eigenvalues:\n{evals[0:n_short]}")

#     if fbr_flag:
#         # compare norms

#         dvr_norms = np.zeros(k_max)
#         for k in range(k_max):
#             dvr_norms[k] = np.sum(evecs[:, k]**2.0, axis=0)

#         fbr_evals, fbr_evecs = eigsh(fbr_H_total, k=k_max, which='SA')

#         fbr_norms = np.zeros(k_max)
#         for k in range(k_max):
#             fbr_norms[k] = np.sum(fbr_evecs[:, k]**2.0, axis=0)

#         # compare norms
#         # print(f"{dvr_norms = }")
#         # print(f"{fbr_norms = }")
#         assert np.allclose(dvr_norms, fbr_norms), 'dvr and fbr norms do not agree'
#         assert np.allclose(dvr_norms, 1.0), 'norm is not 1!'

#         # compare eigenvalues
#         delta_eigvals = evals - fbr_evals
#         print(f"{delta_eigvals = }")
#         print(f"{evals = }")
#         print(f"{fbr_evals = }")
#         import pdb; pdb.set_trace()

#         assert np.allclose(evals, fbr_evals), 'dvr and fbr eigenvalues are different'

#         # end of fbr debug check

#     thermo_props = calculate_thermo_props(evals, basis)

#     delta_E = {
#         'Displaced': evals[1] - evals[0],
#         'Jahn_Teller': evals[2] - evals[0],  # use next gap because Jahn_Teller is degenerate
#     }.get(model, 1.0)
#     assert delta_E != 1.0, 'not a supported model'

#     # shifted_E = (evals[0:n_short] - evals[0])
#     theta = delta_E / kB
#     print(f"{theta = }")
#     # beta = 1. / (kB * theta)
#     # probs = np.exp(-beta * shifted_E)
#     # Z = np.sum(probs)
#     # probs /= Z
#     # print(f"At T={theta}K\nFirst {n_short} probabilities:\n{probs}")

#     if plotting:
#         # figure and axis dictionaries
#         fig_d, ax_d = {}, {}

#         # instantiate the subplots
#         for name in ['EV', 'E', 'CV', 'S', 'A']:
#             fig_d[name], ax_d[name] = plt.subplots()

#         labels = {
#             'Displaced': f"D, gamma={displaced['gamma'][system_index]}",
#             'Jahn_Teller': f"JT, E={jahn_teller['energy'][system_index]} lambda={jahn_teller['lambda'][system_index]}",
#         }.get(model, '')

#         print(labels, f"Theta={delta_E/eV_per_K} K")

#         plot_thermo(ax_d, thermo_props, labels, k_max)
#         label_plots(fig_d, ax_d, model, system_index)
#     #

#     # calculate_distributions(model, system_index, delta_E, evals, evecs, grids, T_list,  k_max, basis)

#     monte_carlo_test(model, system_index, theta, basis)


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

    mc_args = {
        'N_total': int(4e5),
        'N_equilibration': 10,
        'N_skip': int(1e3),
        'Sampling_type': ['Uniform', 'GMD_reduced', 'GMD', 'Direct'][0],
    }

    # profiling_code(model, system_index, mc_args)

    # run
    main(model, system_index, mc_args)
