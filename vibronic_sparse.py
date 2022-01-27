""" vibronic sparse diagonalization
by Pierre-Nicholas Roy, 2022
models from
THE JOURNAL OF CHEMICAL PHYSICS 148, 194110 (2018)
"""

# system imports
import itertools as it
import functools

# third party imports
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import matplotlib as mpl; mpl.use("pdf")  # needed for WSL2 (can also use Agg)
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# local imports


# functions


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
    'w1': .03,
    'w2': .03,
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
        u[v_index] = utemp[i2, a*n1+i1]

    return u


def q1_v(v, lamb_grid1_vec):
    """ act with displaced q1 """
    return np.multiply(lamb_grid1_vec, v)


def q2_v(v, param_times_grid2, basis):
    """ act with displaced q1 """

    # unpack dictionary to local scope for easy readability
    n1, n2 = basis['n1'], basis['n2']

    u = v.copy()  # copy to avoid changing v?

    for i1, i2 in it.product(range(n1), range(n2)):
        a = 0
        u_index = ((a+1)*n1+i1)*n2+i2
        v_index = (a*n1+i1)*n2+i2
        u[u_index] = param_times_grid2[i2] * v[v_index]

        a = 1
        u_index = ((a-1)*n1+i1)*n2+i2
        v_index = (a*n1+i1)*n2+i2
        u[u_index] = param_times_grid2[i2] * v[v_index]

    return u

# -------------------------- plotting functions -------------------------------


def label_plots(figures, axes, model, system_index):
    """ x """

    # remove underscore for matplotlib title
    model_name = model.replace('_', '')

    axes['EV'].set(title=f"E(n) vs n \n{model_name}", xlabel='basis size (n)', ylabel='E(n)/kB (K)')
    axes['E'].set(title=f"<E> vs T \n{model_name}", xlabel='Temperature (K)', ylabel='<E>/kB (K)')
    axes['CV'].set(title=f"Cv vs T \n{model_name}", xlabel='Temperature (K)', ylabel='Cv/kB')
    axes['A'].set(title=f"A vs T \n{model_name}", xlabel='Temperature (K)', ylabel='A/kB (K)')
    axes['S'].set(title=f"S vs T \n{model_name}", xlabel='Temperature (K)', ylabel='S/kB')

    # save to file (leave underscore for file name of plot)
    figures['EV'].savefig(f"E_vs_n_{model}_{system_index}.png")
    figures['E'].savefig(f"E_vs_T_{model}_{system_index}.png")
    figures['CV'].savefig(f"Cv_vs_T_{model}_{system_index}.png")
    figures['A'].savefig(f"A_vs_T_{model}_{system_index}.png")
    figures['S'].savefig(f"S_vs_T_{model}_{system_index}.png")


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


def calculate_distributions(delta_E, evals, evecs, grids, T_list,  kmax, basis):
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
        with open(f"h1_T{t_index}.dat", 'w') as fp:
            fp.write(rho1_filedata)

    def write_h2_to_file(t_index, rho2):

        h2 = np.zeros(n2, float)
        rho2_filedata = ''

        for i2 in range(n2):
            h2[i2] = rho2[i2]

            # multiply by Gauss-Hermite weight
            grid2_contribution = np.exp(-grid2[i2]**2) / (np.sqrt(np.pi)*T2[0, i2]**2)
            h2[i2] *= grid2_contribution

            # append line to string data
            rho2_filedata += f"{grid2[i2]} {h2[i2]}\n"

        # write to file
        with open(f"h2_T{t_index}.dat", 'w') as fp:
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

        with open(f"h12_T{t_index}.dat", 'w') as fp:
            fp.write(rho12_filedata)

        with open(f"w12_T{t_index}.dat", 'w') as fp:
            fp.write(w12_filedata)

    def write_rhoa_to_file(t_index, rhoa):
        rhoa_filedata = ''

        for a, ap in it.product(range(na), range(na)):

            # save to file
            rhoa_filedata += f"{a} {ap} {rhoa[a, ap]}\n"

        with open(f"a_T{t_index}.dat", 'w') as fp:
            fp.write(rhoa_filedata)

    # the temperatures we will evaluate our distributions at
    temperature_list = [0.1, 1., 2., 5., 10.]

    # initialize the distribution arrays
    rho1 = np.zeros((n1), float)
    rho2 = np.zeros((n2), float)
    rho12 = np.zeros((n1, n2), float)
    w12 = np.zeros((n1, n2), float)
    rhoa = np.zeros((na, na), float)

    # eigenvalues with E0 = 0 in units of eV per Kelvin
    Ei = (evals - evals[0]) / eV_per_K
    temp = np.array(temperature_list) * (delta_E / eV_per_K)

    # reshape for easy broadcasting
    Ei_b = Ei.reshape(1, -1)  # (1, N)
    T_b = temp.reshape(-1, 1)  # (nof_temps, 1)

    # axis 1 is `N` the basis dimension
    Z = np.sum(np.exp(-Ei_b/T_b), axis=1)

    for idx, T_val in enumerate(temperature_list):

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
        rho1 /= Z[idx]
        rho2 /= Z[idx]
        rho12 /= Z[idx]
        rhoa /= Z[idx]

        w12 -= t*eV_per_K*np.log(rho12)

        # write all the distributions to file
        write_h1_to_file(T_val, rho1)
        write_h2_to_file(T_val, rho2)
        write_h12_w12_to_file(T_val, rho12, w12)
        write_rhoa_to_file(T_val, rhoa)

    return

# ----------------------------- DVR + Solve H ---------------------------------


def create_dvr_grid(basis):
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

    # define dimentionless q matrices for each mode (basis sizes could be different)
    qmat1 = q_matrix(n1)
    qmat2 = q_matrix(n2)

    # define the dvr grid
    grid1, T1 = np.linalg.eigh(qmat1)
    grid2, T2 = np.linalg.eigh(qmat2)

    # convert h01 and h02 to the DVR
    h01_dvr = np.dot(np.transpose(T1), np.dot(h01, T1))
    h02_dvr = np.dot(np.transpose(T2), np.dot(h02, T2))

    # stick objects into lists for compact handling
    h_terms = [h01_dvr, h02_dvr]
    q_mats = [qmat1, qmat2]
    grids = [grid1, grid2]
    T_list = [T1, T2]

    return h_terms, q_mats, grids, T_list


def build_full_hamiltonian(N, h_terms, grids, system_index, model, basis):
    """ x """

    # unpack dictionary to local scope for easy readability
    na, n1, n2 = basis['a'], basis['n1'], basis['n2']

    # prepare vectors for fast multiplies
    # allocate memory for Ea_list
    Elist_vec = np.zeros(N, float)
    lamb_grid1_vec = np.zeros(N, float)

    # extract the grids
    grid1, grid2 = grids

    # fill Elist_vec and lamb_grid1_vec with appropriate values
    for a, i1, i2 in it.product(range(na), range(n1), range(n2)):

        index = ((a*n1+i1)*n2+i2)

        q1_sign = [1.0, -1.0][a]

        if model == 'Jahn_Teller':
            Elist_vec[index] = jahn_teller['energy'][system_index]

            param_times_grid2 = jahn_teller['lambda'][system_index]*grid2

            coef = jahn_teller['lambda'][system_index]
            lamb_grid1_vec[index] = q1_sign * coef * grid1[i1]

        if model == 'Displaced':
            Elist_vec[index] = displaced['energy'][a]

            param_times_grid2 = displaced['gamma'][system_index]*grid2

            coef = displaced['lambda']
            lamb_grid1_vec[index] = q1_sign * coef * grid1[i1]

    # satisfy the other arguments of the matvec functions
    Ea_func = functools.partial(Ea_v, Elist_vec=Elist_vec)
    h01_func = functools.partial(h01_v, h01_dvr=h_terms[0], basis=basis)
    h02_func = functools.partial(h02_v, h02_dvr=h_terms[1], basis=basis)
    q1_func = functools.partial(q1_v, lamb_grid1_vec=lamb_grid1_vec)
    q2_func = functools.partial(q2_v, param_times_grid2=param_times_grid2, basis=basis)

    # define LinearOperators to preform sparse operations with
    hEa = LinearOperator((N, N), matvec=Ea_func)
    h01 = LinearOperator((N, N), matvec=h01_func)
    h02 = LinearOperator((N, N), matvec=h02_func)
    hq1 = LinearOperator((N, N), matvec=q1_func)
    hq2 = LinearOperator((N, N), matvec=q2_func)

    H_total = hEa+h01+h02+hq1+hq2

    return H_total


def main(model, system_index, plotting=False):
    """ x """

    # basis sizes (store in dictionary for easy passing to functions)
    n1, n2, na = 10, 10, 2
    basis = {'n1': n1, 'n2': n2, 'a': na}

    # total size of product basis
    N = n1*n2*na
    basis['N'] = N

    # modes only basis size
    # n12 = n1*n2

    h_terms, q_mats, grids, T_list = create_dvr_grid(basis)

    H_total = build_full_hamiltonian(N, h_terms, grids, system_index, model, basis)

    kmax = 100
    # niter = 100

    assert kmax < N, f'The number of requested eigenvalues/vectors {kmax = } must be strictly < the basis size {N = } '

    # diagonalize
    # evals, evecs = eigsh(A_total, k=kmax,which = 'SA', maxiter=niter)
    evals, evecs = eigsh(H_total, k=kmax, which='SA')

    thermo_props = calculate_thermo_props(evals, basis)

    delta_E = {
        'Displaced': evals[1] - evals[0],
        'Jahn_Teller': evals[2] - evals[0],  # use next gap because Jahn_Teller is degenerate
    }.get(model, 1.0)

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

        plot_thermo(ax_d, thermo_props, labels, kmax)
        label_plots(fig_d, ax_d, model, system_index)
    #

    calculate_distributions(delta_E, evals, evecs, grids, T_list,  kmax, basis)


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
    p.strip_dirs().sort_stats("tottime").print_stats(8)
    p.strip_dirs().sort_stats("cumulative").print_stats(8)
    p.strip_dirs().sort_stats("cumulative").print_callees('calculate_distributions')
    # p.strip_dirs().sort_stats("cumulative").print_callers('calculate_distributions')


if (__name__ == "__main__"):

    # choose the model
    model = ['Displaced', 'Jahn_Teller'][1]
    system_index = 5  # 0..5 for Displaced and Jahn-Teller

    plot = True  # if plotting the results

    # profiling_code(model, plot)

    # run
    main(model, system_index, plot)
