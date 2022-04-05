import numpy as np


def main(model, system_index, N_total=10000, N_equilibration=100, N_skip=1, Sampling_type='GMD'):

    # basis sizes (store in dictionary for easy passing to functions)
    na = 2
    nmodes = 2

    if model == 'Displaced':
        w1 = displaced['w1']
        w2 = displaced['w2']

    if model == 'Jahn_Teller':
        w1 = jahn_teller['w1']
        w2 = jahn_teller['w2']

    logfile = open(str(model)+str(system_index)+'_MC.log', 'w')
    logfile.write('Model: '+str(model)+'; System_index: '+str(system_index) + '\n')

    # MC test
    # reference displacement
    da0_1 = 0.
    da0_2 = 0.
    da1_1 = 0.
    da1_2 = 0.

    if model == 'Displaced':
        da0_1 = -displaced['lambda']/w1
        da1_1 = displaced['lambda']/w1

        w1_samp = w1
        w2_samp = w2

        da0_1_samp = -1.*displaced['lambda']/w1_samp
        da1_1_samp = 1.*displaced['lambda']/w1_samp
        # da0_1_samp = 0. # only displace q2
        # da1_1_samp = 0.
        # da0_2_samp = -displaced['gamma'][system_index]/w2_samp
        # da1_2_samp = displaced['gamma'][system_index]/w2_samp
        da0_2_samp = 0.
        da1_2_samp = 0.

        Ea0 = displaced['energy'][0]
        Ea1 = displaced['energy'][1]

        Delta0 = -.5*w1*(da0_1**2)
        Delta1 = -.5*w1*(da1_1**2)

        Delta0_samp = -.5*w1_samp*(da0_1_samp**2)
        Delta1_samp = -.5*w1_samp*(da1_1_samp**2)

        # mode 2 sampling
        Delta0_samp += -.5*w2_samp*(da0_2_samp**2)
        Delta1_samp += -.5*w2_samp*(da1_2_samp**2)

        Ea_tilde0 = Ea0+Delta0
        Ea_tilde1 = Ea1+Delta1

    Ea_uniform = np.zeros(2, float)
    Ea_uniform[0] = displaced['energy'][0]
    Ea_uniform[1] = displaced['energy'][1]

    Ea_tilde = np.zeros(2, float)
    Ea_tilde[0] = Ea_tilde0
    Ea_tilde[1] = Ea_tilde1

    for a in range(na):
        Ea_tilde[a] = Ea_tilde[a]-Ea_tilde[0]

    if model == 'Jahn_Teller':
        da0_1 = -jahn_teller['lambda'][system_index]/w1
        da1_1 = jahn_teller['lambda'][system_index]/w1

    dja = np.zeros((nmodes, na), float)
    dja_samp = np.zeros((nmodes, na), float)

    # mode 1
    dja[0, 0] = da0_1
    dja[0, 1] = da1_1

    # mode 2
    dja[1, 0] = da0_2
    dja[1, 1] = da1_2

    # sampling parameters
    # mode 1
    dja_samp[0, 0] = da0_1_samp
    dja_samp[0, 1] = da1_1_samp
    # mode 2
    dja_samp[1, 0] = da0_2_samp
    dja_samp[1, 1] = da1_2_samp

    Ea_tilde0_samp = Ea0+Delta0_samp
    Ea_tilde1_samp = Ea1+Delta1_samp

    Ea_tilde_samp = np.zeros(2, float)
    Ea_tilde_samp[0] = Ea_tilde0_samp
    Ea_tilde_samp[1] = Ea_tilde1_samp

    for a in range(na):
        Ea_tilde_samp[a] = Ea_tilde_samp[a]-Ea_tilde_samp[0]

    P = 16
    T = 300.
    beta = 1./(kB*T)
    tau = beta/float(P)

    def prepare_probability_distributions():
        """ x """
        # prepare probability distributions
        P = 16
        # D, gamma=0.08 Theta= 226.38994229156003  K model 2
        T = 300.

        beta = 1./(kB*T)
        tau = beta/float(P)

        print('tau = ',tau)
        logfile.write('P = '+str(P)+'\n')
        logfile.write('tau (eV) = '+str(tau)+'\n')
        logfile.write('beta (eV) = '+str(beta)+'\n')

        Prob_a = np.zeros(2, float)
        Prob_e = np.zeros(2, float)
        Prob_a_all=np.zeros(2, float)
        a_norm = 0.
        e_norm = 0.
        a_norm_all = 0.

        for a in range(na):
            Prob_e[a] = np.exp(-tau*Ea_tilde[a])
            Prob_a[a] = np.exp(-beta*(Ea_tilde_samp[a]))
            Prob_a_all[a] = np.exp(-tau*(Ea_tilde_samp[a]))
            #Prob_a[a] = np.exp(-beta*(Ea_uniform[a]))
            a_norm += Prob_a[a]
            e_norm += Prob_e[a]
            a_norm_all += Prob_a_all[a]

        Prob_a = (1./a_norm)*Prob_a
        Prob_e = (1./e_norm)*Prob_e
        Prob_a_all = (1./a_norm_all)*Prob_a_all
        logfile.write('Prob_a '+str(Prob_a)+'\n')
        logfile.write('Prob_a_all '+str(Prob_a_all)+'\n')

        return

    rng = default_rng()

    # d/d tau

    C1 = 1./np.tanh(tau*w1)
    C2 = 1./np.tanh(tau*w2)

    C1_prime = -(w1/np.sinh(tau*w1)**2)
    C2_prime = -(w2/np.sinh(tau*w2)**2)

    S1 = 1./np.sinh(tau*w1)
    S2 = 1./np.sinh(tau*w2)

    S1_prime = -(w1*np.cosh(tau*w1)/(np.sinh(tau*w1)**2))
    S2_prime = -(w2*np.cosh(tau*w2)/(np.sinh(tau*w2)**2))

    # -(a Cosh[a x] Csch[a x]^(3/2))/(2 Sqrt[2 Pi])
    F1 = np.sqrt(S1/2./np.pi)
    F2 = np.sqrt(S2/2./np.pi)

    # these don't get used?
    # F1_prime =- (w1*np.cosh(tau*w1)*(1./np.sinh(tau*w1)**(3/2))/(2.*np.sqrt(2.*np.pi)))
    # F2_prime =- (w2*np.cosh(tau*w2)*(1./np.sinh(tau*w2)**(3/2))/(2.*np.sqrt(2.*np.pi)))

    C1_samp = 1./np.tanh(tau*w1_samp)
    C2_samp = 1./np.tanh(tau*w2_samp)

    S1_samp = 1./np.sinh(tau*w1_samp)
    S2_samp = 1./np.sinh(tau*w2_samp)

    # these don't get used?
    # F1_samp = np.sqrt(S1_samp/2./np.pi)
    # F2_samp = np.sqrt(S2_samp/2./np.pi)

    Bmat = build_b_matrix(P)

    mean1 = np.zeros(P, float)
    mean2 = np.zeros(P, float)
    cov1inv = np.zeros((P, P), float)
    cov2inv = np.zeros((P, P), float)
    for p in range(P):
        cov1inv[p, p] = 2.*C1_samp
        cov2inv[p, p] = 2.*C2_samp
        for pp in range(P):
            cov1inv[p, pp] -= S1_samp*Bmat[p, pp]
            cov2inv[p, pp] -= S2_samp*Bmat[p, pp]

    cov1 = np.linalg.inv(cov1inv)
    cov2 = np.linalg.inv(cov2inv)

    outx1x2 = open('x1x2'+str(model)+str(system_index)+'.dat', 'w')

    # recommended numpy random number initialization
    # initial conditions
    # index=rng.choice(na,p=Prob_a)
    # try all zeros
    # sma initial condition for all methods

    x1old = np.random.multivariate_normal(mean1, cov1)
    x2old = np.random.multivariate_normal(mean2, cov2)
    index = 0
    a_old = rng.choice(na, P, p=Prob_a_all)

    for p in range(P):
        a_old[p] = 0
        x1old[p] = 0.
        x2old[p] = 0.

    q1 = x1old+dja_samp[0, index]
    q2 = x2old+dja_samp[1, index]

    pi_1 = np.exp(-.5*(np.dot(x1old, np.dot(cov1inv, x1old))))
    pi_2 = np.exp(-.5*(np.dot(x2old, np.dot(cov2inv, x2old))))

    wa_rhoa_old = pi_1*pi_2
    wa_rhoa_old *= np.exp(-beta*Ea_tilde_samp[index])

    wa_rhoa_old_all = np.exp(-.5*(np.dot(x1old, np.dot(cov1inv, x1old)))-.5*(np.dot(x2old, np.dot(cov2inv, x2old))))

    # modified above for mode bead sampling
    for p in range(P):
        wa_rhoa_old_all *= (-tau*Ea_tilde_samp[a_old[p]])

    # ----------------------------- x ---------------------------------

    o_matrix = calculate_o_matrix()
    m_matrix = calculate_m_matrix()

    # ----------------------------- Metropolis-Hastings loop ---------------------------------

    # compute the initial g value for the MCMH loop
    g_old = np.identity(na)
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