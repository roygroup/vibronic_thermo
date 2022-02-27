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
def main(model, system_index,N_total=10000,N_equilibration=100,N_skip=1):
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

        da0_1_samp=0. # only displace q2
        da1_1_samp=0.

        da0_2_samp=-displaced['gamma'][system_index]/w2_samp
        da1_2_samp=displaced['gamma'][system_index]/w2_samp

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
        Ea_tilde[a]=Ea_tilde[a]-Ea_tilde0
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
        Ea_tilde_samp[a]=Ea_tilde_samp[a]-Ea_tilde0_samp

    P=8
    # D, gamma=0.08 Theta= 226.38994229156003  K model 2
    T=300.

    beta=1./(kB*T)
    tau=beta/float(P)

    print('tau = ',tau)
    logfile.write('P = '+str(P)+'\n')
    logfile.write('tau (eV) = '+str(tau)+'\n')
    logfile.write('beta (eV) = '+str(beta)+'\n')

    Prob_a=np.zeros(2,float)
    a_norm=0.
    for a in range(na):
        #Prob_a[a]=np.exp(-beta*(Ea_tilde_samp[a]))
        Prob_a[a]=np.exp(-beta*(Ea_uniform[a]))
        a_norm+=Prob_a[a]
    Prob_a=(1./a_norm)*Prob_a
    logfile.write(str(Prob_a)+'\n')

    rng = default_rng()

    C1=1./np.tanh(tau*w1)
    S1=1./np.sinh(tau*w1)
    C2=1./np.tanh(tau*w2)
    S2=1./np.sinh(tau*w2)
    F1=np.sqrt(S1/2./np.pi)
    F2=np.sqrt(S2/2./np.pi)

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
    wa_rhoa_old=np.exp(-beta*Ea_tilde_samp[index])*((F1_samp*F2_samp)**P)*pi_1*pi_2

#initial condition for uniform sampling

    for p in range(P):
        q1[p]=0.
        q2[p]=0.

# calculate g without trace
    # build O matrix
    Omat=np.zeros((P,2,2),float)
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
    Omat=(F1*F2)*Omat
 # build M matrix
    Mmat=np.zeros((P,na,na),float)
    Vmat=np.zeros((na,na),float)
    Vmat_diag=np.zeros((na,na),float)

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
        Vmatp=np.dot(Vvec,np.dot(Vmat_diag,np.transpose(Vvec)))
        for a in range(na):
            for ap in range(na):
                Mmat[p,a,ap]=Vmatp[a,ap]
 # build g
 # sum_a' O(R,R',a,a')_daa' . M(R',a',a'')= O(R,R',a,a)M(R',a,a'')
    g_old=np.zeros((na,na),float)
    g_old=np.dot(Omat[0],Mmat[0])
    for p in range(1,P):
        gp=np.dot(g_old,np.dot(Omat[p],Mmat[p]))
        for a in range(na):
            for ap in range(na):
                g_old[a,ap]=gp[a,ap]

    accept=0

    dq1 = 1.
    dq2 = 1.
    q1_new=np.zeros(P,float)
    q2_new=np.zeros(P,float)

    step_count = 0
    for step in range(N_total):  

        index_new=rng.choice(na,p=Prob_a)

        x1_new = np.random.multivariate_normal(mean1, cov1)
        x2_new = np.random.multivariate_normal(mean2, cov2)

        q1_new=x1_new+dja_samp[0,index_new]
        q2_new=x2_new+dja_samp[1,index_new]

# uniform sampling
        for p in range(P):
            q1_new[p]=q1[p]
            q2_new[p]=q2[p]

        p=np.random.randint(0,P)

        #for p in range(P):
        n=np.random.randint(0,2)
        if n==0:
            q1_new[p]=q1[p]+dq1*(2.*np.random.rand()-1.)
        if n==1:
            q2_new[p]=q2[p]+dq2*(2.*np.random.rand()-1.)

        pi_1=np.exp(-.5*(np.dot(x1_new,np.dot(cov1inv,x1_new))))
        pi_2=np.exp(-.5*(np.dot(x2_new,np.dot(cov2inv,x2_new))))

        wa_rhoa_new=np.exp(-beta*Ea_tilde_samp[index_new])*((F1_samp*F2_samp)**P)*pi_1*pi_2

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
            Mmat[p]=np.dot(Vvec,np.dot(Vmat_diag,np.transpose(Vvec)))
 # build g
 # sum_a' O(R,R',a,a')_daa' . M(R',a',a'')= O(R,R',a,a)M(R',a,a'')
        g_new=np.dot(Omat_new[0],Mmat[0])
        for p in range(1,P):
            g_new=np.dot(g_new,np.dot(Omat_new[p],Mmat[p]))
#        ratio=g_new[index_new,index_new]/g_old[index,index]*wa_rhoa_old/wa_rhoa_new
#        ratio=np.trace(g_new)/np.trace(g_old)*wa_rhoa_old/wa_rhoa_new
        ratio=np.trace(g_new)/np.trace(g_old)
        #print(ratio)
        if (ratio >= rng.random()):
            accept+=1
            for p in range(P):
                q1[p]=q1_new[p]
                q2[p]=q2_new[p]
            index=index_new
            for a in range(na):
                for ap in range(na):
                    g_old[a,ap]=g_new[a,ap]
            wa_rhoa_old=wa_rhoa_new

        if step>N_equilibration and step%N_skip==0:
            for p in range(P):
                step_count+=1
                outx1x2.write(str(step_count)+' '+str(q1[p])+' '+str(q2[p])+' '+str(index)+' '+str(ratio)+'\n')

    logfile.write('MC acceptance ratio = '+str(accept/N_total)+'\n')
    logfile.close()

if (__name__ == "__main__"):

    # choose the model
    model = ['Displaced', 'Jahn_Teller'][0]
    system_index = 5 # 0..5 for Displaced and Jahn-Teller
    # run
    system_index=int(sys.argv[1])
    main(model, system_index,N_total=400000,N_equilibration=10,N_skip=1)

