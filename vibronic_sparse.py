""" vibronic sparse diagonalization
by Pierre-Nicholas Roy, 2022
models from
THE JOURNAL OF CHEMICAL PHYSICS 148, 194110 (2018)
"""

# system imports
import sys
import os
from os.path import join, abspath, dirname
import itertools as it
import functools

# third party imports
import numpy as np
# from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import matplotlib as mpl; mpl.use("pdf")  # needed for WSL2 (can also use Agg)
import matplotlib.pyplot as plt
# import matplotlib.tri as tri

# local imports

# ----------------------------- I/O Functions ---------------------------------
output_dir = abspath(join(dirname(__file__), 'output'))
os.makedirs(output_dir, exist_ok=True)  # make sure the output directory exists


def make_figure_path(plot_name):
    """ Factorize out construction of the default figure path.
    Allows for easier modification.
    """
    thermo_dir = join(output_dir, 'thermo')
    os.makedirs(thermo_dir, exist_ok=True)  # make sure the output directory exists

    path = join(thermo_dir, plot_name)

    return path


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

# ----------------------- LinearOperator functions ----------------------------


def Ea_v(v, Elist_vec):
    """ act with diagonal Ea term """
    u = np.multiply(Elist_vec, v)
    return u


def h01_v(v, h01_dvr, basis):
    """ act with  h01 term """

    # unpack dictionary to local scope for easy readability
    na, n1, n2 = basis['a'], basis['n1'], basis['n2']

    u = v.copy()  # copy to avoid changing v?

    vtemp = np.zeros((n1, n2*na), float)
    utemp = np.zeros((n1, n2*na), float)

    # flatten v into a matrix
    for a, i2, i1 in it.product(range(na), range(n2), range(n1)):
        v_index = (a*n1+i1)*n2+i2
        vtemp[i1, a*n2+i2] = v[v_index]

    utemp = np.matmul(h01_dvr, vtemp)

    # unflatten utemp
    for a, i1, i2 in it.product(range(na), range(n1), range(n2)):
        u_index = (a*n1+i1)*n2+i2
        u[u_index] = utemp[i1, a*n2+i2]

    return u


def h02_v(v, h02_dvr, basis):
    """ act with  h02 term
    optimize with blas
    """

    # unpack dictionary to local scope for easy readability
    na, n1, n2 = basis['a'], basis['n1'], basis['n2']

    u = v.copy()  # copy to avoid changing v?

    vtemp = np.zeros((n2, n1*na), float)
    utemp = np.zeros((n2, n1*na), float)

    # flatten v into a matrix
    for a, i2, i1 in it.product(range(na), range(n2), range(n1)):
        v_index = (a*n1+i1)*n2+i2
        vtemp[i2, a*n1+i1] = v[v_index]

    # use blas through dot?
    # utemp=np.dot(h0D2_dvr,vtemp)
    utemp = np.matmul(h02_dvr, vtemp)

    # unflatten utemp
    for a, i1, i2 in it.product(range(na), range(n1), range(n2)):
        u_index = (a*n1+i1)*n2+i2
        u[u_index] = utemp[i2, a*n1+i1]

    return u


def q1_v(v, lamb_grid1_vec):
    """ act with displaced q1 """
    return np.multiply(lamb_grid1_vec, v)


def q2_v(v, scaled_grid2, basis):
    """ act with displaced q2 """

    # unpack dictionary to local scope for easy readability
    n1, n2 = basis['n1'], basis['n2']

    u = v.copy()  # copy to avoid changing v?

    for i1, i2 in it.product(range(n1), range(n2)):
        a = 0
        u_index_a0 = ((a+1)*n1+i1)*n2+i2
        v_index_a0 = (a*n1+i1)*n2+i2
        u[u_index_a0] = scaled_grid2[i2] * v[v_index_a0]

        a = 1
        u_index_a1 = ((a-1)*n1+i1)*n2+i2
        v_index_a1 = (a*n1+i1)*n2+i2
        u[u_index_a1] = scaled_grid2[i2] * v[v_index_a1]

    return u


def fbr_h0_v(v, h0_fbr_terms, basis):
    """ act with h01 term """

    # unpack dictionary to local scope for easy readability
    na, n1, n2 = basis['a'], basis['n1'], basis['n2']

    u = v.copy()  # copy to avoid changing v

    h01, h02 = h0_fbr_terms

    for a, i1, i2 in it.product(range(na), range(n1), range(n2)):
        index = (a*n1+i1)*n2+i2
        u[index] = (h01[i1] + h02[i2])*v[index]

    return u


def fbr_q1_v(v, scaled_qmat1, basis):
    """
    <a|q1|a'> . v[a']

    | +q1  0  | |v0|     |q1.v1|
    | 0   -q1 | |v1|  =  |q1.v0|
    """
    # unpack dictionary to local scope for easy readability
    n1, n2 = basis['n1'], basis['n2']

    u = v.copy()  # copy to avoid changing v?

    left_column = np.zeros((n1, n2), float)
    right_column = np.zeros((n1, n2), float)

    """ plus contribution
    This represents the left COLUMN (a=0) of the matrix
    """
    for i2, i1 in it.product(range(n2), range(n1)):
        v_index_a0 = (0*n1+i1)*n2+i2
        left_column[i1, i2] = +v[v_index_a0]
    left_column = np.matmul(scaled_qmat1, left_column)

    """ minus contribution
    This represents the right COLUMN (a=1) of the matrix
    """
    for i2, i1 in it.product(range(n2), range(n1)):
        v_index_a1 = (1*n1+i1)*n2+i2
        right_column[i1, i2] = -v[v_index_a1]
    right_column = np.matmul(scaled_qmat1, right_column)

    # glue it all together
    for i1, i2 in it.product(range(n1), range(n2)):

        """ a=0 ROW
        we pick the a=0 COLUMN and a=0 ROW of the matrix
        giving us the (0, 0) element which is +q1
        """
        u_index_a0 = (0*n1+i1)*n2+i2
        u[u_index_a0] = left_column[i1, i2]

        """ a=1 ROW
        we pick the a=1 COLUMN and a=1 ROW of the matrix
        giving us the (1, 1) element which is -q1
        """
        u_index_a1 = (1*n1+i1)*n2+i2
        u[u_index_a1] = right_column[i1, i2]

    return u


def fbr_q2_v(v_vector, scaled_qmat2, basis):
    """
    <a|q2|a'> . v[a']

    | 0  q2 | |v0|     |q2.v1|
    | q2 0  | |v1|  =  |q2.v0|
    """

    # unpack dictionary to local scope for easy readability
    n1, n2 = basis['n1'], basis['n2']

    u_vector = v_vector.copy()  # return value

    # temporary arrays
    left_column = np.zeros((n2, n1), float)
    right_column = np.zeros((n2, n1), float)

    """ a=0 contribution
    This represents the left COLUMN (a=0) of the matrix
    """
    for i2, i1 in it.product(range(n2), range(n1)):
        v_index_a0 = (0*n1+i1)*n2+i2
        left_column[i2, i1] = v_vector[v_index_a0]
    left_column = np.matmul(scaled_qmat2, left_column)

    """ a=1 contribution
    This represents the right COLUMN (a=1) of the matrix
    """
    for i2, i1 in it.product(range(n2), range(n1)):
        v_index_a1 = (1*n1+i1)*n2+i2
        right_column[i2, i1] = v_vector[v_index_a1]
    right_column = np.matmul(scaled_qmat2, right_column)

    # glue it all together
    for i1, i2 in it.product(range(n1), range(n2)):

        """ a=0 ROW
        we pick the a=1 COLUMN and a=0 ROW of the matrix
        giving us the (0, 1) element which is q2
        """
        u_index_a0 = (0*n1+i1)*n2+i2
        u_vector[u_index_a0] = right_column[i2, i1]

        """ a=1 ROW
        we pick the a=0 COLUMN and a=1 ROW of the matrix
        giving us the (1, 0) element which is q2
        """
        u_index_a1 = (1*n1+i1)*n2+i2
        u_vector[u_index_a1] = left_column[i2, i1]

    return u_vector

# -------------------------- plotting functions -------------------------------


def label_plots(figures, axes, model, system_index):
    """ x """

    # since the range is 0 -> 5 but in reality we want to plot 1 -> 6
    system_index += 1

    # remove underscore for matplotlib title
    model_name = model.replace('_', '')

    axes['EV'].set(title=f"E(n) vs n \n{model_name}", xlabel='basis size (n)', ylabel='E(n)/kB (K)')
    axes['E'].set(title=f"<E> vs T \n{model_name}", xlabel='Temperature (K)', ylabel='<E>/kB (K)')
    axes['CV'].set(title=f"Cv vs T \n{model_name}", xlabel='Temperature (K)', ylabel='Cv/kB')
    axes['A'].set(title=f"A vs T \n{model_name}", xlabel='Temperature (K)', ylabel='A/kB (K)')
    axes['S'].set(title=f"S vs T \n{model_name}", xlabel='Temperature (K)', ylabel='S/kB')

    # save to file (leave underscore for file name of plot)
    figures['EV'].savefig(make_figure_path(f"E_vs_n_{model}_{system_index}.png"))
    figures['E'].savefig(make_figure_path(f"E_vs_T_{model}_{system_index}.png"))
    figures['CV'].savefig(make_figure_path(f"Cv_vs_T_{model}_{system_index}.png"))
    figures['A'].savefig(make_figure_path(f"A_vs_T_{model}_{system_index}.png"))
    figures['S'].savefig(make_figure_path(f"S_vs_T_{model}_{system_index}.png"))


def plot_thermo(ax_d, thermo, labels, kmax):
    """ plot thermodynamic values """

    # plot and label
    x = [i for i in range(kmax)]
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

# ------------------------------ Statistics -----------------------------------


def calculate_thermo_props(eig_vals, basis, nof_temps=1000):
    """ x """

    # choose temperatures between 0.1 and 10 times the characteristic Theta=delta_E/eV_per_K
    if model == 'Displaced':
        delta_E = eig_vals[1] - eig_vals[0]

    if model == 'Jahn_Teller':
        delta_E = eig_vals[2] - eig_vals[0]  # use next gap because Jahn_Teller is degenerate

    # choose temperatures between 0.1 and 10 time the characteristic Theta=delta_E/eV_per_K
    Tmin, Tmax = 1.0, 100.0

    deltaT = (Tmax - Tmin) / float(nof_temps)

    # temperature values
    T = np.arange(start=Tmin, stop=Tmax, step=deltaT)

    # eigenvalues with E0 = 0 in units of eV per Kelvin
    Ei = (eig_vals - eig_vals[0]) / eV_per_K

    # reshape for easy broadcasting
    Ei_b = Ei.reshape(1, -1)  # (1, N)
    T_b = T.reshape(-1, 1)  # (nof_temps, 1)

    # estimators, Z, <E>, <E^2> (axis 1 is `N` the basis dimension)
    Z = np.sum(np.exp(-Ei_b/T_b), axis=1)
    E = np.sum(np.exp(-Ei_b/T_b)*Ei, axis=1)
    E2 = np.sum(np.exp(-Ei_b/T_b)*Ei*Ei, axis=1)

    # normalize
    E /= Z
    E2 /= Z

    # remaining estimators: Cv, A and S
    Cv = (E2 - E**2) / T**2
    A = -T*np.log(Z)
    S = (E - A) / T

    """ state probabilities
    (we reshape Z to add a dummy dimension to allow for broadcasting)
    `Probs` has shape (nT, N)
    """
    Probs = np.exp(-Ei_b/T_b) / Z.reshape(-1, 1)

    thermo_dictionary = {
        'nof temperatures': nof_temps,
        'T': T, 'Ei': Ei,
        'Z': Z, '<E>': E, '<E^2>': E2,
        'Cv': Cv, 'A': A, 'S': S,
        'P_i': Probs
    }

    return thermo_dictionary


def calculate_distributions(model, system_index, delta_E, evals, evecs, grids, T_list,  kmax, basis):
    """ calculate distributions for each temperature
    """

    # unpack dictionary to local scope for easy readability
    na, n1, n2 = basis['a'], basis['n1'], basis['n2']

    # extract the grids
    grid1, grid2 = grids

    # extract the vectors
    T1, T2 = T_list

    def write_h1_to_file(t_index, rho1):
        h1 = np.zeros(n1, float)
        rho1_filedata = ''

        for i1 in range(n1):
            h1[i1] = rho1[i1]

            # multiply by Gauss-Hermite weight
            grid1_contribution = np.exp(-grid1[i1]**2) / (np.sqrt(np.pi)*T1[0, i1]**2)
            h1[i1] *= grid1_contribution

            # append line to string data
            rho1_filedata += f"{grid1[i1]} {h1[i1]}\n"

        # write to file
        path = make_output_data_path(
            f"{model.lower()}_{system_index+1}"
            f"_h1_T{t_index}.dat"
        )
        with open(path, 'w') as fp:
            fp.write(rho1_filedata)

    def write_h2_to_file(t_index, rho2):

        h2 = np.zeros(n2, float)
        rho2_filedata = ''

        # import pdb; pdb.set_trace()
        for i2 in range(n2):
            h2[i2] = rho2[i2]

            # multiply by Gauss-Hermite weight
            grid2_contribution = np.exp(-grid2[i2]**2) / (np.sqrt(np.pi)*T2[0, i2]**2)
            h2[i2] *= grid2_contribution

            # append line to string data
            rho2_filedata += f"{grid2[i2]} {h2[i2]}\n"
            # pdb.set_trace()

        # write to file
        path = make_output_data_path(
            f"{model.lower()}_{system_index+1}"
            f"_h2_T{t_index}.dat"
        )
        with open(path, 'w') as fp:
            fp.write(rho2_filedata)

    def write_h12_w12_to_file(t_index, rho12, w12):
        rho12_filedata = ''
        w12_filedata = ''

        for i1, i2 in it.product(range(n1), range(n2)):

            # multiply by Gauss-Hermite weight
            grid2_contribution = np.exp(-grid2[i2]**2) / (np.sqrt(np.pi)*T2[0, i2]**2)
            grid1_contribution = np.exp(-grid1[i1]**2) / (np.sqrt(np.pi)*T1[0, i1]**2)
            rho12[i1, i2] *= grid2_contribution * grid1_contribution

            # append line to string data
            rho12_filedata += f"{grid1[i1]} {grid2[i2]} {rho12[i1, i2]}\n"

            # append line to string data
            w12_filedata += f"{grid1[i1]} {grid2[i2]} {w12[i1, i2]}\n"

        path = make_output_data_path(
            f"{model.lower()}_{system_index+1}"
            f"_h12_T{t_index}.dat"
        )
        with open(path, 'w') as fp:
            fp.write(rho12_filedata)

        path = make_output_data_path(
            f"{model.lower()}_{system_index+1}"
            f"_w12_T{t_index}.dat"
        )
        with open(path, 'w') as fp:
            fp.write(w12_filedata)

    def write_rhoa_to_file(t_index, rhoa):
        """ x """

        rhoa_filedata = ''
        for a, ap in it.product(range(na), range(na)):
            rhoa_filedata += f"{a} {ap} {rhoa[a, ap]}\n"

        # save to file
        path = make_output_data_path(
            f"{model.lower()}_{system_index+1}"
            f"_rhoa_T{t_index}.dat"
        )
        with open(path, 'w') as fp:
            fp.write(rhoa_filedata)

    # the temperatures we will evaluate our distributions at
    temperature_list = [0.1, 1., 2., 5., 10.]

    # eigenvalues with E0 = 0 in units of eV per Kelvin
    Ei = (evals - evals[0]) / eV_per_K
    temp = np.array(temperature_list) * (delta_E / eV_per_K)

    # reshape for easy broadcasting
    Ei_b = Ei.reshape(1, -1)  # (1, N)
    T_b = temp.reshape(-1, 1)  # (nof_temps, 1)

    # axis 1 is `N` the basis dimension
    Z = np.sum(np.exp(-Ei_b/T_b), axis=1)

    for idx, T_val in enumerate(temperature_list):

        # initialize the distribution arrays
        rho1 = np.zeros((n1), float)
        rho2 = np.zeros((n2), float)
        rho12 = np.zeros((n1, n2), float)
        w12 = np.zeros((n1, n2), float)
        rhoa = np.zeros((na, na), float)

        t = T_val * (delta_E / eV_per_K)

        exponent = np.exp(-Ei/t)

        for a, ap, i1, i2 in it.product(range(na), range(na), range(n1), range(n2)):
            flattened_index = (a*n1+i1)*n2+i2
            flattened_index2 = (ap*n1+i1)*n2+i2

            rhoa[a, ap] += np.sum(
                evecs[flattened_index, :] * evecs[flattened_index2, :] * exponent
            )

        for i1, i2, a in it.product(range(n1), range(n2), range(na)):
            flattened_index = (a*n1+i1)*n2+i2

            rho1[i1] += np.sum(
                (evecs[flattened_index, :]**2) * exponent
            )

            rho12[i1, i2] += np.sum(
                (evecs[flattened_index, :]**2) * exponent
            )

        for i2, i1, a in it.product(range(n2), range(n1), range(na)):
            flattened_index = (a*n1+i1)*n2+i2
            rho2[i2] += np.sum(
                (evecs[flattened_index, :]**2) * exponent
            )

        # normalize the distributions
        rho1 /= Z[idx]  # rho(q1)
        rho2 /= Z[idx]  # rho(q2)
        rho12 /= Z[idx]  # rho(q1, q2)
        rhoa /= Z[idx]  # rho(a)

        w12 -= t*eV_per_K*np.log(rho12)

        # write all the distributions to file
        write_h1_to_file(T_val, rho1)
        write_h2_to_file(T_val, rho2)
        write_h12_w12_to_file(T_val, rho12, w12)
        write_rhoa_to_file(T_val, rhoa)

    return

# ----------------------------- DVR + Solve H ---------------------------------


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


def create_harmonic_matrices(basis):
    """ x """

    # unpack dictionary to local scope for easy readability
    n1, n2 = basis['n1'], basis['n2']

    # allocate memory for the Hamiltonian terms
    h01 = np.zeros((n1, n1), float)  # diagonal matrix
    h02 = np.zeros((n2, n2), float)  # diagonal matrix

    if model == 'Displaced':
        w1 = displaced['w1']
        w2 = displaced['w2']

    if model == 'Jahn_Teller':
        w1 = jahn_teller['w1']
        w2 = jahn_teller['w2']

    # initialize these matrices
    for i1 in range(n1):
        h01[i1, i1] = w1*(float(i1)+.5)

    for i2 in range(n2):
        h02[i2, i2] = w2*(float(i2)+.5)

    return [h01, h02]


def create_dvr_grid(h01, h02, basis):
    """ x """

    # unpack dictionary to local scope for easy readability
    n1, n2 = basis['n1'], basis['n2']

    # define dimensionless q matrices for each mode (basis sizes could be different)
    qmat1 = q_matrix(n1)
    qmat2 = q_matrix(n2)

    # define the dvr grid
    grid1, T1 = np.linalg.eigh(qmat1)
    grid2, T2 = np.linalg.eigh(qmat2)

    # convert h01 and h02 to the DVR
    h01_dvr = np.dot(np.transpose(T1), np.dot(h01, T1))
    h02_dvr = np.dot(np.transpose(T2), np.dot(h02, T2))

    # debugging
    # print(f"{h02_dvr =}"); import pdb; pdb.set_trace()

    # stick objects into lists for compact handling
    dvr_terms = [h01_dvr, h02_dvr]
    q_mats = [qmat1, qmat2]
    grids = [grid1, grid2]
    T_list = [T1, T2]

    return dvr_terms, q_mats, grids, T_list


def build_full_hamiltonian(N, dvr_h_terms, fbr_h_terms, grids, q_mats, system_index, model, basis, fbr_flag=False):
    """ x """

    # unpack dictionary to local scope for easy readability
    na, n1, n2 = basis['a'], basis['n1'], basis['n2']

    # prepare vectors for fast multiplies
    # allocate memory for Ea_list
    Elist_vec = np.zeros(N, float)
    lamb_grid1_vec = np.zeros(N, float)

    # extract the grids
    grid_q1, grid_q2 = grids

    # fill Elist_vec and lamb_grid1_vec with appropriate values
    for a, i1, i2 in it.product(range(na), range(n1), range(n2)):

        index = ((a*n1+i1)*n2+i2)

        q1_sign = [1.0, -1.0][a]

        if model == 'Jahn_Teller':
            Elist_vec[index] = jahn_teller['energy'][system_index]

            coef = jahn_teller['lambda'][system_index]
            lamb_grid1_vec[index] = q1_sign * coef * grid_q1[i1]

        if model == 'Displaced':
            Elist_vec[index] = displaced['energy'][a]

            coef = displaced['lambda']
            lamb_grid1_vec[index] = q1_sign * coef * grid_q1[i1]

    # prepare the q2 parameters for the LinearOperators
    if model == 'Jahn_Teller':
        scaled_grid2 = jahn_teller['lambda'][system_index] * grid_q2

    if model == 'Displaced':
        scaled_grid2 = displaced['gamma'][system_index] * grid_q2

    # satisfy the other arguments of the matvec functions
    Ea_func = functools.partial(Ea_v, Elist_vec=Elist_vec)
    h01_func = functools.partial(h01_v, h01_dvr=dvr_h_terms[0], basis=basis)
    h02_func = functools.partial(h02_v, h02_dvr=dvr_h_terms[1], basis=basis)
    q1_func = functools.partial(q1_v, lamb_grid1_vec=lamb_grid1_vec)
    q2_func = functools.partial(q2_v, scaled_grid2=scaled_grid2, basis=basis)

    # define LinearOperators to preform sparse operations with
    hEa = LinearOperator((N, N), matvec=Ea_func)
    h01 = LinearOperator((N, N), matvec=h01_func)
    h02 = LinearOperator((N, N), matvec=h02_func)
    hq1 = LinearOperator((N, N), matvec=q1_func)
    hq2 = LinearOperator((N, N), matvec=q2_func)

    # DVR Hamiltonian
    dvr_H_total = hEa+h01+h02+hq1+hq2

    if fbr_flag:

        # extract the q_mats
        q_mat1, q_mat2 = q_mats

        # prepare the q1 and q2 parameters for the LinearOperator
        if model == 'Jahn_Teller':
            # here both q1 and q2 are proportional to lambda
            scaled_qmat1 = jahn_teller['lambda'][system_index] * q_mat1
            scaled_qmat2 = jahn_teller['lambda'][system_index] * q_mat2

        if model == 'Displaced':
            # note q1 is proportional to lambda but q2 is proportional to gamma
            scaled_qmat1 = displaced['lambda'] * q_mat1
            scaled_qmat2 = displaced['gamma'][system_index] * q_mat2

        # satisfy the other arguments of the matvec functions
        h01_func = functools.partial(fbr_h0_v, h0_fbr_terms=fbr_h_terms, basis=basis)
        q1_func = functools.partial(fbr_q1_v, scaled_qmat1=scaled_qmat1, basis=basis)
        q2_func = functools.partial(fbr_q2_v, scaled_qmat2=scaled_qmat2, basis=basis)

        h0_fbr = LinearOperator((N, N), matvec=h01_func)
        hq1_fbr = LinearOperator((N, N), matvec=q1_func)
        hq2_fbr = LinearOperator((N, N), matvec=q2_func)

        # debugging
        # evals, evecs = eigsh(hq2_fbr, k=100, which='SA')
        # print(f"hq2_fbr {evals = }"); import pdb; pdb.set_trace()

        # FBR Hamiltonian
        fbr_H_total = hEa+h0_fbr+hq1_fbr+hq2_fbr

        return dvr_H_total, fbr_H_total

    return dvr_H_total

# --------------------------------- Main --------------------------------------


def main(model, system_index, plotting=False, fbr_flag=False):
    """

    if `plotting` is True then generate plots after all calculations.
    if `fbr_flag` is True then compute H in the Full Basis Representation
        in addition to the DVR.

    """

    # basis sizes (store in dictionary for easy passing to functions)
    n1, n2, na = 10, 10, 2
    basis = {'n1': n1, 'n2': n2, 'a': na}

    # total size of product basis
    N = n1*n2*na
    basis['N'] = N

    # modes only basis size
    # n12 = n1*n2

    h01, h02 = create_harmonic_matrices(basis)

    # create Discrete Variable Representation grid
    dvr_h_terms, q_mats, grids, T_list = create_dvr_grid(h01, h02, basis)

    if fbr_flag:
        # create Full Basis Representation h terms
        fbr_h01 = np.diag(h01)
        fbr_h02 = np.diag(h02)
        fbr_h_terms = [fbr_h01, fbr_h02]
    else:
        fbr_h_terms = [None, None]

    args = (
        N, dvr_h_terms, fbr_h_terms, grids, q_mats,
        system_index, model, basis, fbr_flag
    )
    if not fbr_flag:
        dvr_H_total = build_full_hamiltonian(*args)
    else:
        dvr_H_total, fbr_H_total = build_full_hamiltonian(*args)

    k_max = 100

    assert k_max < N, (
        f'The number of requested eigenvalues/vectors {k_max = } '
        f'must be strictly < the basis size {N = }'
    )

    # diagonalize
    # niter = 100
    # evals, evecs = eigsh(A_total, k=k_max, which = 'SA', maxiter=niter)
    evals, evecs = eigsh(dvr_H_total, k=k_max, which='SA')

    if fbr_flag:
        # compare norms

        dvr_norms = np.zeros(k_max)
        for k in range(k_max):
            dvr_norms[k] = np.sum(evecs[:, k]**2.0, axis=0)

        fbr_evals, fbr_evecs = eigsh(fbr_H_total, k=k_max, which='SA')

        fbr_norms = np.zeros(k_max)
        for k in range(k_max):
            fbr_norms[k] = np.sum(fbr_evecs[:, k]**2.0, axis=0)

        # compare norms
        # print(f"{dvr_norms = }")
        # print(f"{fbr_norms = }")
        assert np.allclose(dvr_norms, fbr_norms), 'dvr and fbr norms do not agree'
        assert np.allclose(dvr_norms, 1.0), 'norm is not 1!'

        # compare eigenvalues
        delta_eigvals = evals - fbr_evals
        print(f"{delta_eigvals = }")
        print(f"{evals = }")
        print(f"{fbr_evals = }")
        import pdb; pdb.set_trace()

        assert np.allclose(evals, fbr_evals), 'dvr and fbr eigenvalues are different'

        # end of fbr debug check

    thermo_props = calculate_thermo_props(evals, basis)

    delta_E = {
        'Displaced': evals[1] - evals[0],
        'Jahn_Teller': evals[2] - evals[0],  # use next gap because Jahn_Teller is degenerate
    }.get(model, 1.0)

    assert delta_E != 1.0, 'not a supported model'

    if plotting:
        # figure and axis dictionaries
        fig_d, ax_d = {}, {}

        # instantiate the subplots
        for name in ['EV', 'E', 'CV', 'S', 'A']:
            fig_d[name], ax_d[name] = plt.subplots()

        labels = {
            'Displaced': f"D, gamma={displaced['gamma'][system_index]}",
            'Jahn_Teller': f"JT, E={jahn_teller['energy'][system_index]} lambda={jahn_teller['lambda'][system_index]}",
        }.get(model, '')

        print(labels, f"Theta={delta_E/eV_per_K} K")

        plot_thermo(ax_d, thermo_props, labels, k_max)
        label_plots(fig_d, ax_d, model, system_index)
    #

    calculate_distributions(model, system_index, delta_E, evals, evecs, grids, T_list,  k_max, basis)


def profiling_code(model, system_index, plot):
    """ simple profiling """

    import cProfile
    import pstats

    filename = 'cProfile_output'
    cProfile.runctx(
        'main(model, system_index, plot)',
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

    plot = True  # if plotting the results

    # profiling_code(model, system_index, plot)

    # run
    assert 0 <= system_index <= 5, f'Currently only takes 0,1,2,3,4, or 5, not {system_index}'
    main(model, system_index, plot)
