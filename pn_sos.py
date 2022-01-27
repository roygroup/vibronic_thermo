# Thermodynamics of models from
# THE JOURNAL OF CHEMICAL PHYSICS 148, 194110 (2018)

# system imports
import itertools as it

# third party imports
import numpy as np
from numpy import newaxis as NEW
import matplotlib as mpl; mpl.use("pdf")  # needed for WSL2
import matplotlib.pyplot as plt

# local imports

def q_matrix(basis_size):
    """ Build continuous dimension matrix """
    matrix = np.zeros((basis_size, basis_size), float)

    for i, ip in it.product(range(basis_size), repeat=2):
        if ip == (i+1):
            matrix[i, ip] = np.sqrt(float(i) + 1.0)

        if ip == (i-1):
            matrix[i, ip] = np.sqrt(float(i))

        matrix[i, ip] /= np.sqrt(2.0)

    return matrix


def delta(i, j):
    """ Kronecker delta function"""
    return 1.0 if i == j else 0.0


# constants
eV_per_K = 8.617333262e-5
kB = eV_per_K

# displaced model parameters (all in eV)
g1 = 0.00
g2 = 0.04
g3 = 0.08
g4m3 = 0.09
g4m2 = 0.1
g4m1 = 0.11
g4 = 0.12
g4p1 = 0.13
g4p2 = 0.14
g4p3 = 0.15
g5 = 0.16
g6 = 0.20

displaced = {
    'energy': [0.0996, 0.1996],
    'gamma': [g1, g2, g3, g4, g5, g6],
    'lambda': 0.072,
    'w1': 0.02,
    'w2': 0.04,
}


# finding three critical points
# gmin=.11
# gmax=.13
# ng=10
# dg=(gmax-gmin)/ng
# g_list=[]
# for i in range(ng):
#    g_list.append(gmin+i*dg)

# Jahn Teller system (all in eV)
jahn_teller = {
    'energy': [0.02999, 0.00333, 0.07666, 0.20999, 0.39667, 0.63135, 0.03, 0.03],
    'lambda': [0.00, 0.04, 0.08, 0.12, 0.16, 0.20],
    'w1': .03,
    'w2': .03,
}


def label_plots(figures, axes, model):
    """ x """

    # remove underscore for matplotlib title
    model_name = model.replace('_', '')

    axes['EV'].set(title=f"E(n) vs n \n{model_name}", xlabel='basis size (n)', ylabel='E(n)/kB (K)')
    axes['E'].set(title=f"<E> vs T \n{model_name}", xlabel='Temperature (K)', ylabel='<E>/kB (K)')
    axes['CV'].set(title=f"Cv vs T \n{model_name}", xlabel='Temperature (K)', ylabel='Cv/kB')
    axes['A'].set(title=f"A vs T \n{model_name}", xlabel='Temperature (K)', ylabel='A/kB (K)')
    axes['S'].set(title=f"S vs T \n{model_name}", xlabel='Temperature (K)', ylabel='S/kB')

    # save to file (leave underscore for file name of plot)
    figures['EV'].savefig(f"E_vs_n_{model}.png")
    figures['E'].savefig(f"E_vs_T_{model}.png")
    figures['CV'].savefig(f"Cv_vs_T_{model}.png")
    figures['A'].savefig(f"A_vs_T_{model}.png")
    figures['S'].savefig(f"S_vs_T_{model}.png")


def plot_thermo(ax_d, thermo, labels, basis):
    """ plot thermodynamic values """

    # plot and label
    x = [i for i in range(basis['N'])]
    ax_d['EV'].plot(x, thermo['Ei'], label=labels)
    ax_d['EV'].legend(loc="upper left")

    ax_d['E'].plot(thermo['T'], thermo['<E>'], label=labels)
    ax_d['E'].legend(loc="upper left")

    ax_d['CV'].plot(thermo['T'], thermo['Cv'], label=labels)
    ax_d['CV'].legend(loc="upper right")

    ax_d['A'].plot(thermo['T'], thermo['A'], label=labels)
    ax_d['A'].legend(loc="lower left")

    ax_d['S'].plot(thermo['T'], thermo['S'], label=labels)
    ax_d['S'].legend(loc="upper right")


def create_h0_matrix(model, basis):
    """ x """

    # unpack dictionary to local scope for easy readability
    nq1, nq2 = basis['n1'], basis['n2']

    # modes only basis size
    n12 = nq1 * nq2

    # allocate memory for the h0 Hamiltonian
    if model == 'Displaced':
        h0_matrix = np.zeros((n12), float)  # diagonal matrix

        for i1, i2 in it.product(range(nq1), range(nq2)):
            # harmonic oscillator omega * 1/2 + n
            index = i1*nq2 + i2
            h0_matrix[index] = displaced['w1']*(float(i1)+.5) + displaced['w2']*(float(i2)+.5)

    elif model == 'Jahn_Teller':
        h0_matrix = np.zeros((n12), float)  # diagonal matrix

    elif model == 'Jahn_Teller':
        for i1, i2 in it.product(range(nq1), range(nq2)):
            # harmonic oscillator omega * 1/2 + n
            index = i1*nq2 + i2
            h0_matrix[index] = jahn_teller['w1']*(float(i1)+.5) + jahn_teller['w2']*(float(i2)+.5)

    return h0_matrix


def calculate_thermo_props(eig_vals, basis, nof_temps=300):
    """ x """

    # choose temperatures between 0.1 and 10 times the characteristic Theta=delta_E/eV_per_K
    delta_E = eig_vals[1] - eig_vals[0]
    theta = delta_E/eV_per_K

    old_Tmin = 1.0*theta
    old_Tmax = 3.0*theta

    Tmin, Tmax = 0.1, 300.0

    deltaT = (Tmax - Tmin) / float(nof_temps)

    # initialize arrays
    # Z = np.zeros(nof_temps, float)
    # E = np.zeros(nof_temps, float)
    # E2 = np.zeros(nof_temps, float)
    # Cv = np.zeros(nof_temps, float)
    # A = np.zeros(nof_temps, float)
    # S = np.zeros(nof_temps, float)

    # temperature values
    T = np.arange(start=Tmin, stop=Tmax, step=deltaT)

    # eigenvalues with E0 = 0 in units of eV per Kelvin
    Ei = (eig_vals - eig_vals[0]) / eV_per_K

    # reshape for easy broadcasting
    Ei2 = Ei.reshape(1, -1)  # (1, N)
    T2 = T.reshape(-1, 1)  # (nof_temps, 1)

    # estimators, Z, <E>, <E^2>
    Z = np.sum(np.exp(-Ei2/T2), axis=1)
    E = np.sum(np.exp(-Ei2/T2)*Ei, axis=1)
    E2 = np.sum(np.exp(-Ei2/T2)*Ei*Ei, axis=1)

    # normalize
    E /= Z
    E2 /= Z

    # remaining estimators: Cv, A and S
    Cv = (E2 - E**2) / T**2
    A = -T*np.log(Z)
    S = (E - A) / T

    thermo_dictionary = {
        'nof temperatures': nof_temps,
        'T': T, 'Ei': Ei,
        'Z': Z, '<E>': E, '<E^2>': E2,
        'Cv': Cv, 'A': A, 'S': S
    }

    return thermo_dictionary


def build_full_hamiltonian(H, h0_matrix, q1mat, q2mat, s_index, model, basis):
    """ x """

    # unpack dictionary to local scope for easy readability
    nof_a, nq1, nq2 = basis['a'], basis['n1'], basis['n2']

    if model == 'Displaced':
        lambda_value = displaced['lambda']
        gamma_value = displaced['gamma'][s_index]

    if model == 'Jahn_Teller':
        lambda_value = jahn_teller['lambda'][s_index]
        energy = jahn_teller['energy'][s_index]

    for a in range(nof_a):

        q1_sign = [1.0, -1.0][a]

        if model == 'Displaced':
            energy = displaced['energy'][a]

        for i1, i2 in it.product(range(nq1), range(nq2)):

            for ap, i1p, i2p in it.product(range(nof_a), range(nq1), range(nq2)):

                if model == 'Displaced':

                    energy_offset = (energy + h0_matrix[i1*nq2+i2]) * delta(a, ap) * delta(i1, i1p) * delta(i2, i2p)
                    q1_contribution = q1_sign * lambda_value * q1mat[i1, i1p] * delta(a, ap) * delta(i2, i2p)
                    q2_contribution = gamma_value * q2mat[i2, i2p] * delta(i1, i1p) * (1. - delta(a, ap))

                if model == 'Jahn_Teller':

                    energy_offset = (energy + h0_matrix[i1*nq2+i2]) * delta(a, ap) * delta(i1, i1p) * delta(i2, i2p)
                    q1_contribution = q1_sign * lambda_value * q1mat[i1, i1p] * delta(a, ap) * delta(i2, i2p)
                    q2_contribution = lambda_value * q2mat[i2, i2p] * delta(i1, i1p) * (1. - delta(a, ap))

                index = ((a*nq1+i1)*nq2+i2, (ap*nq1+i1p)*nq2+i2p)

                H[index] = energy_offset + q1_contribution + q2_contribution
    return


def main(model, plotting=False):
    """ x """

    if plotting:
        # figure and axis dictionaries
        fig_d, ax_d = {}, {}

        # instantiate the subplots
        for name in ['EV', 'E', 'CV', 'S', 'A']:
            fig_d[name], ax_d[name] = plt.subplots()

    nof_systems = 6

    # basis sizes (store in dictionary for easy passing to functions)
    n1, n2, a = 20, 20, 2
    basis = {'n1': n1, 'n2': n2, 'a': a}

    # total size of product basis
    N = n1*n2*a
    basis['N'] = N

    # allocate memory for full Hamiltonian
    H = np.zeros((N, N), float)

    h0_matrix = create_h0_matrix(model, basis)

    # define dimensionless q matrices for each mode (basis sizes could be different)
    q1_matrix = q_matrix(n1)
    q2_matrix = q_matrix(n2)

    # loop over the 6 different lambda/gamma values
    for s_index in range(nof_systems):

        # fill H
        build_full_hamiltonian(H, h0_matrix, q1_matrix, q2_matrix, s_index, model, basis)

        # diagonalize
        if False:  # if we need the eigenvectors
            eig_vals, eig_vecs = np.linalg.eigh(H)
        else:
            eig_vals = np.linalg.eigvalsh(H)

        delta_E = eig_vals[1] - eig_vals[0]

        if model == 'Displaced':
            gamma_value = displaced['gamma'][s_index]
            labels = f"D, gamma = {gamma_value}"

        if model == 'Jahn_Teller':
            lambda_value = jahn_teller['lambda'][s_index]
            energy = jahn_teller['energy'][s_index]
            labels = f"JT, E = {energy} lambda = {lambda_value}"

        print(labels, f"Theta = {delta_E/eV_per_K}(K)")

        thermo_props = calculate_thermo_props(eig_vals, basis)

        if plotting:
            plot_thermo(ax_d, thermo_props, labels, basis)

    if plotting:
        label_plots(fig_d, ax_d, model)


def profiling_code(model, plot):
    """ simple profiling """

    import cProfile
    import pstats

    filename = 'cProfile_output'
    cProfile.runctx(
        'main(model, plot)',
        globals(),
        locals(),
        filename
    )
    p = pstats.Stats(filename)
    p.strip_dirs().sort_stats("tottime").print_stats(6)
    p.strip_dirs().sort_stats("cumulative").print_stats(20)
    # p.strip_dirs().sort_stats("cumulative").print_stats('calculate', 15)


if (__name__ == "__main__"):

    # choose the model
    model = ['Displaced', 'Jahn_Teller'][1]

    plot = True  # if plotting the results

    # profiling_code(model, plot)

    # run
    main(model, plot)
