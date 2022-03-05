# vibronic sparse diagonalization
# by Pierre-Nicholas Roy, 2022
# models from
# THE JOURNAL OF CHEMICAL PHYSICS 148, 194110 (2018)
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
# functions
def q_matrix(size):
    # q coordinate in the harmonic oscillator eigenstate basis
    qmat=np.zeros((size,size),float)
    for i in range(size):
        for ip in range(size):
            if ip==(i+1):
                qmat[i,ip]=np.sqrt(float(i)+1.)
            if ip==(i-1):
                qmat[i,ip]=np.sqrt(float(i))
            qmat[i,ip]/=np.sqrt(2.)
    return qmat
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
def main(model, system_index,distributions=False):
    def Ea_v(v): # act with diagonal Ea term
        N =len(v)
        u=np.multiply(Elist_vec,v)
        return u
    def h01_v(v): # act with  h01 term
        N =len(v)   
        u=v.copy()  
        vtemp=np.zeros((n1,n2*na),float)
        utemp=np.zeros((n1,n2*na),float)
        for a in range(na):
            for i2 in range(n2):
                for i1 in range(n1):
                    vtemp[i1,a*n2+i2]=v[((a*n1+i1)*n2+i2)]
        utemp=np.matmul(h01_dvr,vtemp)
        for a in range(na):
            for i1 in range(n1):
                for i2 in range(n2):
                    u[((a*n1+i1)*n2+i2)]=utemp[i1,a*n2+i2]
        return u
    def h02_v(v): # act with  h02 term
        #  optimize with blas
        N =len(v)
        u=v.copy()  
        vtemp=np.zeros((n2,n1*na),float)
        utemp=np.zeros((n2,n1*na),float)
        for a in range(na):
            for i2 in range(n2):
                for i1 in range(n1):
                    vtemp[i2,a*n1+i1]=v[((a*n1+i1)*n2+i2)]
    # use blas through dot?
    #utemp=np.dot(h0D2_dvr,vtemp)
        utemp=np.matmul(h02_dvr,vtemp)
        for a in range(na):
            for i1 in range(n1):
                for i2 in range(n2):
                    u[((a*n1+i1)*n2+i2)]=utemp[i2,a*n1+i1]    
        return u
    def h0_fbr_v(v): # act with  h01 term
        N =len(v)   
        u=v.copy()  
        for a in range(na):
            for i1 in range(n1):
                for i2 in range(n2):
                    u[((a*n1+i1)*n2+i2)]=(h01_fbr[i1]+h02_fbr[i2])*v[((a*n1+i1)*n2+i2)]
        return u    
    def q1_v(v): #act with displaced q1
        N =len(v)
        u=np.multiply(lamb_grid1_vec,v)
        return u
    def q1_fbr_v(v):
        N =len(v)   
        u=v.copy()  
        vtemp=np.zeros((n1,n2),float)
        utemp=np.zeros((n1,n2),float)
        a=0 # plus contribution
        for i2 in range(n2):
            for i1 in range(n1):
                vtemp[i1,i2]=v[((a*n1+i1)*n2+i2)]
        utemp=np.matmul(qmat1,vtemp)
        a=1 # minus contribution
        for i2 in range(n2):
            for i1 in range(n1):
                vtemp[i1,i2]=-v[((a*n1+i1)*n2+i2)]
        vtemp=np.matmul(qmat1,vtemp)
        for i1 in range(n1):
            for i2 in range(n2):
                a=0
                u[((a*n1+i1)*n2+i2)]=utemp[i1,i2]
                a=1
                u[((a*n1+i1)*n2+i2)]=vtemp[i1,i2]
        return u

    def q2_fbr_v(v):
    #
    # <a|q2|a'> . v[a']
    # 
    #  | 0  q2 | |v0|     |q2.v1|
    #  | q2 0  | |v1|   = |q2.v0|
    # 
        N =len(v)   
        u=v.copy()  
        vtemp=np.zeros((n2,n1),float)
        utemp=np.zeros((n2,n1),float)
        a=0 
        for i2 in range(n2):
            for i1 in range(n1):
                vtemp[i2,i1]=v[((a*n1+i1)*n2+i2)]
        utemp=np.matmul(qmat2,vtemp)
        a=1 
        for i2 in range(n2):
            for i1 in range(n1):
                vtemp[i2,i1]=v[((a*n1+i1)*n2+i2)]
        vtemp=np.matmul(qmat2,vtemp)
        for i1 in range(n1):
            for i2 in range(n2):
                a=0
                u[((a*n1+i1)*n2+i2)]=vtemp[i2,i1]
                a=1
                u[((a*n1+i1)*n2+i2)]=utemp[i2,i1]
        return u

    def q2_v(v): #act with displaced q1
    #
    # <a|q2|a'> . v[a']
    # 
    #  | 0  q2 | |v0|     |q2.v1|
    #  | q2 0  | |v1|   = |q2.v0|
    # 
        N =len(v)
        u=v.copy()
        for i1 in range(n1):
            for i2 in range(n2):
                a=0
                u[(((a+1)*n1+i1)*n2+i2)]=param_times_grid2[i2]*v[((a*n1+i1)*n2+i2)]
                a=1
                u[(((a-1)*n1+i1)*n2+i2)]=param_times_grid2[i2]*v[((a*n1+i1)*n2+i2)]
        return u
 # basis sizes (store in dictionary for easy passing to functions)
    n1, n2, na =80, 80, 2
    nmodes=2
    basis = {'n1': n1, 'n2': n2, 'a': na}

    # total size of product basis
    N = n1*n2*na
    basis['N'] = N
    # modes only basis size
    n12=n1*n2
    # allocate memory for the hamiltonian terms
    h01=np.zeros((n1,n1),float) # diagonal matrix
    h02=np.zeros((n2,n2),float) # diagonal matrix
    h01_fbr=np.zeros(n1,float) 
    h02_fbr=np.zeros(n2,float) 

    if model=='Displaced':
        w1=displaced['w1']
        w2=displaced['w2']
    if model=='Jahn_Teller':
        w1=jahn_teller['w1']
        w2=jahn_teller['w2']
    # initialize these matrices
    for i1 in range(n1):        
        h01[i1,i1]=w1*(float(i1)+.5)
        h01_fbr[i1]=h01[i1,i1]
    for i2 in range(n2):
        h02[i2,i2]=w2*(float(i2)+.5)
        h02_fbr[i2]=h02[i2,i2]

    # define dimentionless q matrices for each mode (basis sizes could be different)
    qmat1=q_matrix(n1)
    qmat2=q_matrix(n2)

    # define the dvr grid
    grid1, T1 = np.linalg.eigh(qmat1)
    grid2, T2 = np.linalg.eigh(qmat2)
    param_times_grid2=grid2.copy()

    if model=='Displaced':
        param_times_grid2=displaced['gamma'][system_index]*grid2
        qmat2=displaced['gamma'][system_index]*qmat2
        qmat1=displaced['lambda']*qmat1
    if model=='Jahn_Teller':
        param_times_grid2=jahn_teller['lambda'][system_index]*grid2
        qmat2=jahn_teller['lambda'][system_index]*qmat2
        qmat1=jahn_teller['lambda'][system_index]*qmat1

    # convert h01 and h02 to the DVR
    h01_dvr=np.dot(np.transpose(T1),np.dot(h01,T1))
    h02_dvr=np.dot(np.transpose(T2),np.dot(h02,T2))
    #prepare vectors for fast multiplies
    # allocate memory for Ea_list
    Elist_vec=np.zeros(N,float)
    lamb_grid1_vec=np.zeros(N,float)
    
    for a in range(na):
        for i1 in range(n1):
            for i2 in range(n2):
                if model=='Jahn_Teller':
                    Elist_vec[((a*n1+i1)*n2+i2)]=jahn_teller['energy'][system_index]
                    if (a==0):
                        lamb_grid1_vec[((a*n1+i1)*n2+i2)]=jahn_teller['lambda'][system_index]*grid1[i1]
                    if (a==1):
                        lamb_grid1_vec[((a*n1+i1)*n2+i2)]=(-jahn_teller['lambda'][system_index])*grid1[i1]
                if model=='Displaced':
                    Elist_vec[((a*n1+i1)*n2+i2)]=displaced['energy'][a]
                    if (a==0):
                        lamb_grid1_vec[((a*n1+i1)*n2+i2)]=displaced['lambda']*grid1[i1]
                    if (a==1):
                        lamb_grid1_vec[((a*n1+i1)*n2+i2)]=(-displaced['lambda'])*grid1[i1]


    hEa = LinearOperator((N,N), matvec=Ea_v)
    h01 = LinearOperator((N,N), matvec=h01_v)
    h02 = LinearOperator((N,N), matvec=h02_v)
    hq1 = LinearOperator((N,N), matvec=q1_v)
    hq2 = LinearOperator((N,N), matvec=q2_v)
    h0_fbr = LinearOperator((N,N), matvec=h0_fbr_v)
    hq1_fbr = LinearOperator((N,N), matvec=q1_fbr_v)
    hq2_fbr = LinearOperator((N,N), matvec=q2_fbr_v)

# DVR Hamiltonian
    H_total=hEa+h01+h02+hq1+hq2
    H_fbr_total=hEa+h0_fbr+hq1_fbr+hq2_fbr

    kmax=100
    niter=100
    #evals, evecs = eigsh(A_total, k=kmax,which = 'SA', maxiter=niter)
    evals, evecs = eigsh(H_total, k=kmax,which = 'SA')


    # test norm
    '''
    for k in  range(kmax):
        norm=0
        for i in range(N):
            norm+=evecs[i,k]**2 
        print(norm) 
    '''
    # test of fbr calculation
    '''
    evals_fbr,evecs_fbr = eigsh(H_fbr_total, k=kmax,which = 'SA')
    for i in range(kmax):   
        print(i,evals[i]-evals_fbr[i])
    '''
    # it works

    if model=='Displaced':
        delta_E=evals[1]-evals[0]
    if model=='Jahn_Teller':
        delta_E=evals[2]-evals[0] # use next gap because Jahn_Teller is degenerate

    Theta=delta_E/eV_per_K

    # choose temperatures between 0.1 and 10 time the characteristic Theta=delta_E/eV_per_K
    Tmin=.1*Theta
    Tmax=3.*Theta
    nT=1000 # number of temperature values
    deltaT=(Tmax-Tmin)/float(nT)
    T=np.zeros(nT,float)
    Probs=np.zeros((nT,kmax),float)
    Z=np.zeros(nT,float)
    E=np.zeros(nT,float)
    E2=np.zeros(nT,float)
    Cv=np.zeros(nT,float)
    A=np.zeros(nT,float)
    S=np.zeros(nT,float)
    for t in range(nT):
        T[t]=Tmin+t*deltaT
    # estimators, <E>, Cv, S, and Z
        Z[t]=0.
        for i in range(kmax):
            Ei=(evals[i]-evals[0])/eV_per_K
            Z[t]+=np.exp(-Ei/T[t])
            E[t]+=np.exp(-Ei/T[t])*Ei
            E2[t]+=np.exp(-Ei/T[t])*Ei*Ei
        E[t]/=Z[t]
        E2[t]/=Z[t]
        Cv[t]=(E2[t]-E[t]**2)/(T[t]**2)
        A[t]=-T[t]*np.log(Z[t])
        S[t]=(E[t]-A[t])/T[t]
        for i in range(kmax):
            Ei=(evals[i]-evals[0])/eV_per_K
            Probs[t,i]=np.exp(-Ei/T[t])/Z[t]

    if model=='Displaced':
        labels='D, gamma='+str(displaced['gamma'][system_index])
    if model=='Jahn_Teller':
        labels='JT, E='+str(jahn_teller['energy'][system_index])+' lambda='+str(jahn_teller['lambda'][system_index])

    logfile=open(str(model)+str(system_index)+'.log','w')
    logfile.write('Model: '+str(model)+'; System_index: '+str(system_index)+ '\n')
    logfile.write('Basis: n1 n2 na = '+str(n1)+' '+str(n2)+' '+str(na)+'\n')
    logfile.write(' Theta(n1,n2,na) = '+str(Theta)+' K'+'\n')
    logfile.write('Prob(n,Theta)= '+str((np.exp(-(evals[0:kmax]-evals[0])/(kB*5.*Theta))))+'\n')

    fig_EV, ax_EV = plt.subplots()
    fig_E, ax_E = plt.subplots()
    fig_Cv, ax_Cv = plt.subplots()
    fig_S, ax_S = plt.subplots()
    fig_A, ax_A = plt.subplots()
    ax_EV.plot([i for i in range(kmax)],(evals-evals[0])/eV_per_K,'o',label=labels)
    ax_E.plot(T,E,label=labels)
    ax_Cv.plot(T,Cv,label=labels)
    ax_Cv.plot([Theta,Theta],[0,np.max(Cv)],'--')
    ax_A.plot(T,A,label=labels)
    ax_S.plot(T,S,label=labels)
    ax_EV.set(title='E(n) vs n '+str(model),xlabel='n',ylabel='E(n)/kB (K)')
    ax_E.set(title='<E> vs T '+str(model),xlabel='T (K)',ylabel='<E>/kB (K)')
    ax_Cv.set(title='Cv vs T '+str(model),xlabel='T (K)',ylabel='Cv/kB ')
    ax_A.set(title='A vs T '+str(model),xlabel='T (K)',ylabel='A/kB (K)')
    ax_S.set(title='S vs T '+str(model),xlabel='T (K)',ylabel='S/kB')
    fig_EV.savefig('Evsn_'+str(model)+str(system_index)+'.png')
    fig_E.savefig('EvsT_'+str(model)+str(system_index)+'.png')
    fig_Cv.savefig('CvvsT_'+str(model)+str(system_index)+'.png')
    fig_A.savefig('AvsT_'+str(model)+str(system_index)+'.png')
    fig_S.savefig('SvsT_'+str(model)+str(system_index)+'.png')  

   
    if distributions==True:
    # calculate distributions for each temperature
        Tlist=[0.1, 1., 2., 5., 10.,300./Theta,400./Theta]
        for tindex in Tlist:
            rho1=np.zeros((n1),float)
            rho2=np.zeros((n2),float)
            rho12=np.zeros((n1,n2),float)
            rho1a=np.zeros((n1,na),float)
            rho2a=np.zeros((n1,na),float)
            w12=np.zeros((n1,n2),float)
            rhoa=np.zeros((na,na),float)
        #output files
            rho1_out=open('h1_T'+str(tindex)+str(model)+str(system_index)+'.dat','w')
            rho2_out=open('h2_T'+str(tindex)+str(model)+str(system_index)+'.dat','w')
            rho12_out=open('h12_T'+str(tindex)+str(model)+str(system_index)+'.dat','w')
            rho1a_out=open('h1a_T'+str(tindex)+str(model)+str(system_index)+'.dat','w')
            rho2a_out=open('h2a_T'+str(tindex)+str(model)+str(system_index)+'.dat','w')
            w12_out=open('w12_T'+str(tindex)+str(model)+str(system_index)+'.dat','w')
            rhoa_out=open('a_T'+str(tindex)+str(model)+str(system_index)+'.dat','w')
            
            t=tindex*Theta
            Z=0.
            for i in range(kmax):
                Ei=(evals[i]-evals[0])/kB
                Z+=np.exp(-Ei/t)
                for a in range(na):
                    for ap in range(na):
                        for i1 in range(n1):
                            for i2 in range(n2):
                                rhoa[a,ap]+=evecs[(a*n1+i1)*n2+i2,i]*evecs[(ap*n1+i1)*n2+i2,i]*np.exp(-Ei/t)
                for i1 in range(n1):
                    for i2 in range(n2):
                        for a in range(na):
                            sum=(evecs[(a*n1+i1)*n2+i2,i]**2)*np.exp(-Ei/t)
                            rho1[i1]+=sum
                            rho2[i2]+=sum
                            rho12[i1,i2]+=sum
                            rho1a[i1,a]+=sum
                            rho2a[i2,a]+=sum

            rho1=(1./Z)*rho1
            rho2=(1./Z)*rho2
            rho12=(1./Z)*rho12
            rho1a=(1./Z)*rho1a
            rho2a=(1./Z)*rho2a
            rhoa=(1./Z)*rhoa
            w12=-t*eV_per_K*np.log(rho12)

        #grid1, T1    
        #convert to grid
            for i1 in range(n1):
                h1=rho1[i1]
            # multiply by gauss hermite weight
                h1*=np.exp(-grid1[i1]**2)/(np.sqrt(np.pi)*T1[0,i1]**2)
                rho1_out.write(str(grid1[i1])+' '+str(h1)+' '+str(-t*eV_per_K*np.log(h1))+'\n')
    
            for i2 in range(n2):
                h2=rho2[i2]
            # multiply by gauss hermite weight
                h2*=np.exp(-grid2[i2]**2)/(np.sqrt(np.pi)*T2[0,i2]**2)
                rho2_out.write(str(grid2[i2])+' '+str(h2)+' '+str(-t*eV_per_K*np.log(h2))+'\n')

            for i1 in range(n1):
                for i2 in range(n2):
                # multiply by gauss hermite weight
                    rho12[i1,i2]*=((np.exp(-grid2[i2]**2)/(np.sqrt(np.pi)*T2[0,i2]**2))*(np.exp(-grid1[i1]**2)/(np.sqrt(np.pi)*T1[0,i1]**2)))
                    rho12_out.write(str(grid1[i1])+' '+str(grid2[i2])+' '+str(rho12[i1,i2])+'\n')
                    w12_out.write(str(grid1[i1])+' '+str(grid2[i2])+' '+str(w12[i1,i2])+'\n')

            for a in range(na):
                for ap in range(na):
                    rhoa_out.write(str(a)+' '+str(ap)+' '+str(rhoa[a,ap])+'\n')
            for i1 in range(n1):
                for a in range(na):
                    rho1a[i1,a]*=np.exp(-grid1[i1]**2)/(np.sqrt(np.pi)*T1[0,i1]**2)
                rho1a_out.write(str(grid1[i1])+' '+str(rho1a[i1,0])+' '+str(rho1a[i1,1])+'\n')
            for i2 in range(n2):
                for a in range(na):
                    rho2a[i2,a]*=np.exp(-grid2[i2]**2)/(np.sqrt(np.pi)*T2[0,i2]**2)
                rho2a_out.write(str(grid2[i2])+' '+str(rho2a[i2,0])+' '+str(rho2a[i2,1])+'\n')

            rho1_out.close()
            rho2_out.close()
            rho12_out.close()
            rho1a_out.close()
            rho2a_out.close()
            w12_out.close()
            rhoa_out.close()
    logfile.close()

if (__name__ == "__main__"):

    # choose the model
    model = ['Displaced', 'Jahn_Teller'][0]
    system_index = 5 # 0..5 for Displaced and Jahn-Teller
    distributions=True
    # run
    system_index=int(sys.argv[1])
    main(model,system_index,distributions)
