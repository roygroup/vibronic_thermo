# vibronic sparse diagonalization
# by Pierre-Nicholas Roy, 2022
# models from
# THE JOURNAL OF CHEMICAL PHYSICS 148, 194110 (2018)
import numpy as np
from numpy.random import default_rng
import sys
# functions
def B(size):
    Bmat=np.zeros((size,size),float)
    for i in range(size):
        for ip in range(size):
            if ip==(i+1):
                Bmat[i,ip]=1.
            if ip==(i-1):
                Bmat[i,ip]=1.
    Bmat[0,size-1]=1.
    Bmat[size-1,0]=1.
    return Bmat
# constants
eV_per_K=8.617333262e-5
kB=eV_per_K

#displaced model parameters (all in eV)
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
def main(model, system_index,N_total=10000,N_equilibration=100,N_skip=1,Sampling_type='GMD'):
 # basis sizes (store in dictionary for easy passing to functions)
    na =2 
    nmodes=2
    if model=='Displaced':
        w1=displaced['w1']
        w2=displaced['w2']
    if model=='Jahn_Teller':
        w1=jahn_teller['w1']
        w2=jahn_teller['w2']
    logfile=open(str(model)+str(system_index)+'_MC.log','w')
    logfile.write('Model: '+str(model)+'; System_index: '+str(system_index)+ '\n')

    # MC test
    # reference displacement
    da0_1=0.
    da0_2=0.
    da1_1=0.
    da1_2=0.

    if model=='Displaced':
        da0_1=-displaced['lambda']/w1
        da1_1=displaced['lambda']/w1

        w1_samp=w1
        w2_samp=w2

        da0_1_samp=-1.*displaced['lambda']/w1_samp
        da1_1_samp=1.*displaced['lambda']/w1_samp
        #da0_1_samp=0. # only displace q2
        #da1_1_samp=0.
        #da0_2_samp=-displaced['gamma'][system_index]/w2_samp
        #da1_2_samp=displaced['gamma'][system_index]/w2_samp
        da0_2_samp=0.
        da1_2_samp=0.

        Ea0=displaced['energy'][0]
        Ea1=displaced['energy'][1]

        Delta0=-.5*w1*(da0_1**2)
        Delta1=-.5*w1*(da1_1**2)

        Delta0_samp=-.5*w1_samp*(da0_1_samp**2)
        Delta1_samp=-.5*w1_samp*(da1_1_samp**2)

# mode 2 sampling
        Delta0_samp+=-.5*w2_samp*(da0_2_samp**2)
        Delta1_samp+=-.5*w2_samp*(da1_2_samp**2)

        Ea_tilde0=Ea0+Delta0
        Ea_tilde1=Ea1+Delta1
      
    Ea_uniform=np.zeros(2,float)
    Ea_uniform[0]=displaced['energy'][0]
    Ea_uniform[1]=displaced['energy'][1]
    
    Ea_tilde=np.zeros(2,float)
    Ea_tilde[0]=Ea_tilde0
    Ea_tilde[1]=Ea_tilde1
    for a in range(na): 
        Ea_tilde[a]=Ea_tilde[a]-Ea_tilde[0]
    if model=='Jahn_Teller':
        da0_1=-jahn_teller['lambda'][system_index]/w1
        da1_1=jahn_teller['lambda'][system_index]/w1

    dja=np.zeros((nmodes,na),float)
    dja_samp=np.zeros((nmodes,na),float)

    #mode 1
    dja[0,0]=da0_1
    dja[0,1]=da1_1
    #mode 2
    dja[1,0]=da0_2
    dja[1,1]=da1_2
#sampling papameters
    #mode 1
    dja_samp[0,0]=da0_1_samp
    dja_samp[0,1]=da1_1_samp
    #mode 2
    dja_samp[1,0]=da0_2_samp
    dja_samp[1,1]=da1_2_samp

    Ea_tilde0_samp=Ea0+Delta0_samp
    Ea_tilde1_samp=Ea1+Delta1_samp

    Ea_tilde_samp=np.zeros(2,float)
    Ea_tilde_samp[0]=Ea_tilde0_samp
    Ea_tilde_samp[1]=Ea_tilde1_samp
    for a in range(na): 
        Ea_tilde_samp[a]=Ea_tilde_samp[a]-Ea_tilde_samp[0]

    P=64
    # D, gamma=0.08 Theta= 226.38994229156003  K model 2
    T=300.

    beta=1./(kB*T)
    tau=beta/float(P)

    print('tau = ',tau)
    logfile.write('P = '+str(P)+'\n')
    logfile.write('tau (eV) = '+str(tau)+'\n')
    logfile.write('beta (eV) = '+str(beta)+'\n')

    Prob_a=np.zeros(2,float)
    Prob_a_all=np.zeros(2,float)
    a_norm=0.
    a_norm_all=0.
    for a in range(na):
        Prob_a[a]=np.exp(-beta*(Ea_tilde_samp[a]))
        Prob_a_all[a]=np.exp(-tau*(Ea_tilde_samp[a]))
        #Prob_a[a]=np.exp(-beta*(Ea_uniform[a]))
        a_norm+=Prob_a[a]
        a_norm_all+=Prob_a_all[a]
    Prob_a=(1./a_norm)*Prob_a
    Prob_a_all=(1./a_norm_all)*Prob_a_all
    logfile.write('Prob_a '+str(Prob_a)+'\n')
    logfile.write('Prob_a_all '+str(Prob_a_all)+'\n')

    rng = default_rng()

    C1=1./np.tanh(tau*w1)
    # d/d tau
    C1_prime=-(w1/np.sinh(tau*w1)**2)
    S1=1./np.sinh(tau*w1)
    S1_prime=-(w1*np.cosh(tau*w1)/(np.sinh(tau*w1)**2))

    C2=1./np.tanh(tau*w2)
    C2_prime=-(w2/np.sinh(tau*w2)**2)
    S2=1./np.sinh(tau*w2)
    S2_prime=-(w2*np.cosh(tau*w2)/(np.sinh(tau*w2)**2))

    F1=np.sqrt(S1/2./np.pi)
# -(a Cosh[a x] Csch[a x]^(3/2))/(2 Sqrt[2 Pi])
    F1_prime=-(w1*np.cosh(tau*w1)*(1./np.sinh(tau*w1)**(3/2))/(2.*np.sqrt(2.*np.pi)))
    F2=np.sqrt(S2/2./np.pi)
    F2_prime=-(w2*np.cosh(tau*w2)*(1./np.sinh(tau*w2)**(3/2))/(2.*np.sqrt(2.*np.pi)))

    C1_samp=1./np.tanh(tau*w1_samp)
    S1_samp=1./np.sinh(tau*w1_samp)
    C2_samp=1./np.tanh(tau*w2_samp)
    S2_samp=1./np.sinh(tau*w2_samp)
    F1_samp=np.sqrt(S1_samp/2./np.pi)
    F2_samp=np.sqrt(S2_samp/2./np.pi)

    Bmat=B(P)

    mean1 =np.zeros(P,float)
    cov1inv = np.zeros((P,P),float)
    mean2 =np.zeros(P,float)
    cov2inv = np.zeros((P,P),float)
    for p in range(P):
        cov1inv[p,p]=2.*C1_samp
        cov2inv[p,p]=2.*C2_samp
        for pp in range(P):
            cov1inv[p,pp]-=S1_samp*Bmat[p,pp]
            cov2inv[p,pp]-=S2_samp*Bmat[p,pp]

    cov1=np.linalg.inv(cov1inv)
    cov2=np.linalg.inv(cov2inv)

    outx1x2=open('x1x2'+str(model)+str(system_index)+'.dat','w')
    # recommanded numpy random number initialization
    #initial conditions
    index=rng.choice(na,p=Prob_a)

    x1old = np.random.multivariate_normal(mean1, cov1)
    x2old = np.random.multivariate_normal(mean2, cov2)

    q1=x1old+dja_samp[0,index]
    q2=x2old+dja_samp[1,index]

    pi_1=np.exp(-.5*(np.dot(x1old,np.dot(cov1inv,x1old))))
    pi_2=np.exp(-.5*(np.dot(x2old,np.dot(cov2inv,x2old))))

    #wa_rhoa_old=np.exp(-beta*Ea_tilde_samp[index])*((F1_samp*F2_samp)**P)*pi_1*pi_2
    wa_rhoa_old=pi_1*pi_2
    wa_rhoa_old*=np.exp(-beta*Ea_tilde_samp[index])

    wa_rhoa_old_all=np.exp(-.5*(np.dot(x1old,np.dot(cov1inv,x1old)))-.5*(np.dot(x2old,np.dot(cov2inv,x2old))))
    #modified above for mode bead sampling
    a_old=rng.choice(na,P,p=Prob_a_all)
    for p in range(P):
        wa_rhoa_old_all*=(-tau*Ea_tilde_samp[a_old[p]])
    # q from all mode beads
    if Sampling_type=='Direct':
        for p in range(P):
            q1[p]=x1old[p]+dja_samp[0,a_old[p]]
            q2[p]=x2old[p]+dja_samp[1,a_old[p]]

# calculate g without trace
    # build O matrix
    Omat=np.zeros((P,2,2),float)
    Omat_E=np.zeros((P,2,2),float)
    for p in range(P):
        for a in range(na):
            if (p+1)<P:
                x1=q1[p]-dja[0,a]
                x1p=q1[p+1]-dja[0,a]
                x2=q2[p]-dja[1,a]
                x2p=q2[p+1]-dja[1,a]
            else:
                x1=q1[P-1]-dja[0,a]
                x1p=q1[0]-dja[0,a]
                x2=q2[P-1]-dja[1,a]
                x2p=q2[0]-dja[1,a]
            Omat[p,a,a]=np.exp(-tau*Ea_tilde[a])
            Omat[p,a,a]*=np.exp(S1*(x1*x1p)-.5*C1*(x1**2+x1p**2))
            Omat[p,a,a]*=np.exp(S2*(x2*x2p)-.5*C2*(x2**2+x2p**2))

            Omat_E[p,a,a]=Ea_tilde[a]+S1*(x1*x1p)*S1_prime-.5*C1*(x1**2+x1p**2)*C1_prime
            Omat_E[p,a,a]+=S2*(x2*x2p)*S2_prime-.5*C2*(x2**2+x2p**2)*C2_prime

    Omat=(F1*F2)*Omat

    #for p in range(P):
    #    print(p,Omat[p,:])

 # build M matrix
    Mmat=np.zeros((P,na,na),float)
    Mmat_E=np.zeros((P,na,na),float)
    Vmat=np.zeros((na,na),float)
    Vmat_diag=np.zeros((na,na),float)
    Vmat_diag_E=np.zeros((na,na),float)

    for p in range(P):
        if model=='Displaced':
            Vmat[0,1]=displaced['gamma'][system_index]*q2[p]
            Vmat[1,0]=Vmat[0,1]
        if model=='Jahn_Teller':
            Vmat[0,1]=jahn_teller['lambda'][system_index]*q2[p]
            Vmat[1,0]=Vmat[0,1]
        Vval,Vvec=np.linalg.eigh(Vmat)
        for a in range(na):
            Vmat_diag[a,a]=np.exp(-tau*Vval[a])
            Vmat_diag_E[a,a]=Vval[a]

        Vmatp=np.dot(Vvec,np.dot(Vmat_diag,np.transpose(Vvec)))
        Vmatp_E=np.dot(Vvec,np.dot(Vmat_diag_E,np.transpose(Vvec)))
        for a in range(na):
            for ap in range(na):
                Mmat[p,a,ap]=Vmatp[a,ap]
                Mmat_E[p,a,ap]=Vmatp_E[a,ap]
    """ qmin=-10.
    qmax=10.
    Nq=100
    dq=(qmax-qmin)/Nq
    for i in range(Nq):
        qval=qmin+i*dq
        Vmat[0,0]=0.
        Vmat[1,1]=0.
        Vmat[0,1]=displaced['gamma'][system_index]*qval
        Vmat[1,0]=Vmat[0,1]
        Vval,Vvec=np.linalg.eigh(Vmat)
        for a in range(na):
            Vmat_diag[a,a]=np.exp(-tau*Vval[a])
        Vmatp=np.dot(Vvec,np.dot(Vmat_diag,np.transpose(Vvec)))
        print(qval,Vmatp[0,0],Vmatp[1,1],Vmatp[0,1],Vmatp[1,0]) """
 # build g
 # sum_a' O(R,R',a,a')_daa' . M(R',a',a'')= O(R,R',a,a)M(R',a,a'')
    #print(Omat)
    #print(Mmat)
    # for p in range(P):
    #     for a in range(na):
    #         for ap in range(na):
    #             Omat[p,a,ap]+=1.e-16
    #             Mmat[p,a,ap]+=1.e-16

    g_old=np.zeros((na,na),float)
    g_old=np.dot(Omat[0],Mmat[0])
    for p in range(1,P):
        gp=np.dot(g_old,np.dot(Omat[p],Mmat[p]))
        for a in range(na):
            for ap in range(na):
                g_old[a,ap]=gp[a,ap]
    # g calculate for state bead sampling
    g_old_scalar=1.
    for p in range(P):
        if p+1<P:
            g_old_scalar*=((Mmat[p,a_old[p],a_old[p+1]]))*np.exp(-tau*Ea_tilde[a_old[p]]+(S1*(x1*x1p)-.5*C1*(x1**2+x1p**2))+(S2*(x2*x2p)-.5*C2*(x2**2+x2p**2)))
        else:
            g_old_scalar*=((Mmat[P-1,a_old[P-1],a_old[0]]))*np.exp(-tau*Ea_tilde[a_old[P-1]]+(S1*(x1*x1p)-.5*C1*(x1**2+x1p**2))+(S2*(x2*x2p)-.5*C2*(x2**2+x2p**2)))
    accept=0

    dq1 = 1.
    dq2 = 1.
    q1_new=np.zeros(P,float)
    q2_new=np.zeros(P,float)
    a_new=a_old.copy()
    g_new_scalar=g_old_scalar
    wa_rhoa_new_all=wa_rhoa_old_all
    wa_rhoa_new=wa_rhoa_old
    index_new=index

    step_count = 0
    for step in range(N_total):  

        if Sampling_type=='GMD' or Sampling_type=='GMD_reduced':
            index_new=rng.choice(na,p=Prob_a)
            x1_new = np.random.multivariate_normal(mean1, cov1)
            x2_new = np.random.multivariate_normal(mean2, cov2)
            for p in range(P):
                q1_new[p]=x1_new[p]+dja_samp[0,index_new]
                q2_new[p]=x2_new[p]+dja_samp[1,index_new]
            pi_1=np.exp(-.5*(np.dot(x1_new,np.dot(cov1inv,x1_new))))
            pi_2=np.exp(-.5*(np.dot(x2_new,np.dot(cov2inv,x2_new))))
            wa_rhoa_new=np.exp(-beta*Ea_tilde_samp[index_new])*pi_1*pi_2

        elif Sampling_type=='Direct':
            x1_new = np.random.multivariate_normal(mean1, cov1)
            x2_new = np.random.multivariate_normal(mean2, cov2)
            a_new=rng.choice(na,P,p=Prob_a_all)
            for p in range(P):
                q1_new[p]=x1_new[p]+dja_samp[0,a_new[p]]
                q2_new[p]=x2_new[p]+dja_samp[1,a_new[p]]

            wa_rhoa_new_all=np.exp(-.5*(np.dot(x1_new,np.dot(cov1inv,x1_new))))+(-.5*(np.dot(x2_new,np.dot(cov2inv,x2_new))))
            for p in range(P):
                wa_rhoa_new_all*=(-tau*Ea_tilde_samp[a_new[p]])

        elif Sampling_type=='Uniform':
            for p in range(P):
                q1_new[p]=q1[p]
                q2_new[p]=q2[p]
            p=np.random.randint(0,P)
            n=np.random.randint(0,2)
            if n==0:
                q1_new[p]=q1[p]+dq1*(2.*np.random.rand()-1.)
            if n==1:
                q2_new[p]=q2[p]+dq2*(2.*np.random.rand()-1.)

        Omat_new=np.zeros((P,2,2),float)
        for p in range(P):
            for a in range(na):
                if (p+1)<P:
                    x1=q1_new[p]-dja[0,a]
                    x1p=q1_new[p+1]-dja[0,a]
                    x2=q2_new[p]-dja[1,a]
                    x2p=q2_new[p+1]-dja[1,a]
                else:
                    x1=q1_new[P-1]-dja[0,a]
                    x1p=q1_new[0]-dja[0,a]
                    x2=q2_new[P-1]-dja[1,a]
                    x2p=q2_new[0]-dja[1,a]
                Omat_new[p,a,a]=np.exp(-tau*Ea_tilde[a])
                Omat_new[p,a,a]*=np.exp(S1*(x1*x1p)-.5*C1*(x1**2+x1p**2))
                Omat_new[p,a,a]*=np.exp(S2*(x2*x2p)-.5*C2*(x2**2+x2p**2))

        Omat_new=(F1*F2)*Omat_new
 # build M matrix
        Vmat=np.zeros((na,na),float)
        Vmat_diag=np.zeros((na,na),float)
        for p in range(P):
            if model=='Displaced':
                Vmat[0,1]=displaced['gamma'][system_index]*q2_new[p]
                Vmat[1,0]=Vmat[0,1]
            if model=='Jahn_Teller':
                Vmat[0,1]=jahn_teller['lambda'][system_index]*q2_new[p]
                Vmat[1,0]=Vmat[0,1]
            Vval,Vvec=np.linalg.eigh(Vmat)
            for a in range(na):
                Vmat_diag[a,a]=np.exp(-tau*Vval[a])
            Vmatp=np.dot(Vvec,np.dot(Vmat_diag,np.transpose(Vvec)))
            for a in range(na):
                for ap in range(na):
                    Mmat[p,a,ap]=Vmatp[a,ap]
                    if a==ap:
                        Mmat[p,a,ap]=1.
                    else:
                        Mmat[p,a,ap]=0.
        g_new=np.dot(Omat_new[0],Mmat[0])
        for p in range(1,P):
            g_new=np.dot(g_new,np.dot(Omat_new[p],Mmat[p]))
        ratio=1.
        if Sampling_type=='GMD':
            ratio=g_new[index_new,index_new]/g_old[index,index]*wa_rhoa_old/wa_rhoa_new
        if Sampling_type=='GMD_reduced':
            ratio=np.trace(g_new)*wa_rhoa_old/(np.trace(g_old)*wa_rhoa_new)
        if Sampling_type=='Direct':
            g_new_scalar=1.
            for p in range(P):
                if p+1<P:
                    g_new_scalar*=((Mmat[p,a_new[p],a_new[p+1]]))*np.exp(-tau*Ea_tilde[a_new[p]]+(S1*(x1*x1p)-.5*C1*(x1**2+x1p**2))+(S2*(x2*x2p)-.5*C2*(x2**2+x2p**2)))
                else:
                    g_new_scalar*=((Mmat[P-1,a_new[P-1],a_new[0]]))*np.exp(-tau*Ea_tilde[a_new[P-1]]+(S1*(x1*x1p)-.5*C1*(x1**2+x1p**2))+(S2*(x2*x2p)-.5*C2*(x2**2+x2p**2)))
            
            ratio_num=g_new_scalar*wa_rhoa_old_all
            ratio_denom=g_old_scalar*wa_rhoa_new_all
            ratio=ratio_num/ratio_denom
            print(g_old_scalar,g_new_scalar)

        
        if (ratio >= rng.random()):
            accept+=1
            for p in range(P):
                q1[p]=q1_new[p]
                q2[p]=q2_new[p]
                a_old[p]=a_new[p]
            index=index_new
            for a in range(na):
                for ap in range(na):
                    g_old[a,ap]=g_new[a,ap]
            wa_rhoa_old=wa_rhoa_new
            g_old_scalar=g_new_scalar
            wa_rhoa_old_all=wa_rhoa_new_all

        if step>N_equilibration and step%N_skip==0:
            for p in range(P):
                step_count+=1
                outx1x2.write(str(step_count)+' '+str(q1[p])+' '+str(q2[p])+' '+str(index)+' '+str(a_old[p])+' '+str(ratio)+'\n')

    logfile.write('MC acceptance ratio = '+str(accept/N_total)+'\n')
    logfile.close()

if (__name__ == "__main__"):

    # choose the model
    model = ['Displaced', 'Jahn_Teller'][0]
    system_index = 5 # 0..5 for Displaced and Jahn-Teller
    # run
    system_index=int(sys.argv[1])
    main(model, system_index,N_total=400000,N_equilibration=10,N_skip=1,Sampling_type='GMD')
    # GMD_reduced, GME, Direct, Uniform

