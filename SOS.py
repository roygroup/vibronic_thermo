# Thermodynamics of models from
# THE JOURNAL OF CHEMICAL PHYSICS 148, 194110 (2018)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
def q_matrix(size):
    qmat=np.zeros((size,size),float)
    for i in range(size):
        for ip in range(size):
            if ip==(i+1):
                qmat[i,ip]=np.sqrt(float(i)+1.)
            if ip==(i-1):
                qmat[i,ip]=np.sqrt(float(i))
            qmat[i,ip]/=np.sqrt(2.)
    return qmat

def delta(i,j):
    if i==j:
        return 1.
    else:
        return 0.

def mv(v):
    N =len(v)
    u=v.copy()
    u=np.dot(HDVR,v)
    return u

def Ea_v(v): # act with diagonal Ea term
    N =len(v)
    u=v.copy()
    for a in range(na):
        for i1 in range(n1):
            for i2 in range(n2):
                u[((a*n1+i1)*n2+i2)]=Elist[a]*v[((a*n1+i1)*n2+i2)]
    return u
def h01_v(v): # act with  h01 term
    N =len(v)   
    u=v.copy()  
    for a in range(na):
        for i2 in range(n2):
            for i1 in range(n1):
                for i1p in range(n1):
                    u[((a*n1+i1)*n2+i2)]+=h0D1_dvr[i1,i1p]*v[((a*n1+i1p)*n2+i2)]
    return u
def h02_v(v): # act with  h02 term
    #  optimize with blas
    N =len(v)
    u=v.copy()
    for a in range(na):
        for i1 in range(n1):
            for i2 in range(n2):
                for i2p in range(n2):
                    u[((a*n1+i1)*n2+i2)]+=h0D2_dvr[i2,i2p]*v[((a*n1+i1)*n2+i2p)]
    return u
def q1_v(v): #act with displaced q1
    N =len(v)
    u=v.copy()
    for i1 in range(n1):
        for i2 in range(n2):
            a=0
            u[((a*n1+i1)*n2+i2)]=lamb*grid1[i1]*v[((a*n1+i1)*n2+i2)]
            a=1
            u[((a*n1+i1)*n2+i2)]=(-lamb*grid1[i1]*v[((a*n1+i1)*n2+i2)])
    return u
def q2_v(v): #act with displaced q1
    N =len(v)
    u=v.copy()
    for i1 in range(n1):
        for i2 in range(n2):
            a=0
            u[(((a+1)*n1+i1)*n2+i2)]=g_list[s]*grid2[i2]*v[((a*n1+i1)*n2+i2)]
            a=1
            u[(((a-1)*n1+i1)*n2+i2)]=g_list[s]*grid2[i2]*v[((a*n1+i1)*n2+i2)]
    return u

# constants
eV_per_K=8.617333262e-5
kB=eV_per_K
#displaced model
# parameters (all in eV)
Ea=0.0996
Eb=0.1996
Elist=[Ea,Eb]
q1sign=[1.,-1.]
w1=0.02
w2=0.04
lamb=0.072
g1=0.00
g2=0.04
g3=0.08
g4m3=0.09
g4m2=0.1
g4m1=0.11
g4=0.12
g4p1=0.13
g4p2=0.14
g4p3=0.15
g5=0.16
g6=0.20
g_list=[g1,g2,g3,g4,g5,g6]
# finding thge critical point
#gmin=.11
#gmax=.13
#ng=10
#dg=(gmax-gmin)/ng
#g_list=[]
#for i in range(ng):
#    g_list.append(gmin+i*dg)

# Jahn Teller system
EJT=[0.02999,0.00333,0.07666,0.20999,0.39667,0.63135,0.03,0.03]
lambdaJT=[0.00,0.04,0.08,0.12,0.16,0.20]
w1JT=.03
w2JT=.03

nsystems=6

# basis sizes
n1=40
n2=40
na=2
# total size of product basis
N=n1*n2*na
# modes only basis size
n12=n1*n2
# allocate memory for full hamiltonian
H=np.zeros((N,N),float)
HDVR=np.zeros((N,N),float)
# allocate memory for the h0 hamiltonian
h0D=np.zeros((n12),float) # diagonal matrix
h0JT=np.zeros((n12),float) # diagonal matrix

for i1 in range(n1):        
    for i2 in range(n2):
        h0D[i1*n2+i2]=w1*(float(i1)+.5)+w2*(float(i2)+.5)
        h0JT[i1*n2+i2]=w1JT*(float(i1)+.5)+w2JT*(float(i2)+.5)

h0D1=np.zeros((n1,n1),float) # diagonal matrix
h0JT1=np.zeros((n1,n1),float) # diagonal matrix
h0D2=np.zeros((n2,n2),float) # diagonal matrix
h0JT2=np.zeros((n2,n2),float) # diagonal matrix

for i1 in range(n1):        
    h0D1[i1,i1]=w1*(float(i1)+.5)
    h0JT1[i1,i1]=w1JT*(float(i1)+.5)
for i2 in range(n2):
    h0D2[i2,i2]=w2*(float(i2)+.5)
    h0JT2[i2,i2]=w2JT*(float(i2)+.5)

# define dimentionless q matrices for each mode (basis sizes could be different)
qmat1=q_matrix(n1)
qmat2=q_matrix(n2)

grid1, T1 = np.linalg.eigh(qmat1)
grid2, T2 = np.linalg.eigh(qmat2)

# convert h01 and h02 to the DVR
h0D1_dvr=np.dot(np.transpose(T1),np.dot(h0D1,T1))
h0JT1_dvr=np.dot(np.transpose(T1),np.dot(h0JT1,T1))
h0D2_dvr=np.dot(np.transpose(T2),np.dot(h0D2,T2))
h0JT2_dvr=np.dot(np.transpose(T2),np.dot(h0JT2,T2))

#print(qmat1)
# build h0

fig_EV, ax_EV = plt.subplots()
fig_E, ax_E = plt.subplots()
fig_Cv, ax_Cv = plt.subplots()
fig_S, ax_S = plt.subplots()
fig_A, ax_A = plt.subplots()
fig_rho1, ax_rho1 = plt.subplots()
fig_rho2, ax_rho2 = plt.subplots()
fig_rho12, ax_rho12 = plt.subplots()
fig_rhoa, ax_rhoa = plt.subplots()

# choice of model
model='Displaced'
#model='Jahn_Teller'

count=0
#build full Hamiltonian
for s in range(nsystems):
    for a in range(na):
        for i1 in range(n1):
            for i2 in range(n2):
                for ap in range(na):
                    for i1p in range(n1):         
                        for i2p in range(n2):
                            H[(a*n1+i1)*n2+i2,(ap*n1+i1p)*n2+i2p]=0.
                            HDVR[(a*n1+i1)*n2+i2,(ap*n1+i1p)*n2+i2p]=0.
                            #displaced H
                            if model=='Displaced':
                            #    H[(a*n1+i1)*n2+i2,(ap*n1+i1p)*n2+i2p]=(Elist[a]+h0D[i1*n2+i2])*delta(a,ap)*delta(i1,i1p)*delta(i2,i2p)\
                            #    +q1sign[a]*lamb*qmat1[i1,i1p]*delta(a,ap)*delta(i2,i2p)\
                            #    +g_list[s]*qmat2[i2,i2p]*delta(i1,i1p)*(1.-delta(a,ap))

                                HDVR[(a*n1+i1)*n2+i2,(ap*n1+i1p)*n2+i2p]=(Elist[a])*delta(a,ap)*delta(i1,i1p)*delta(i2,i2p)\
                                +h0D1_dvr[i1,i1p]*delta(a,ap)*delta(i2,i2p)\
                                +h0D2_dvr[i2,i2p]*delta(a,ap)*delta(i1,i1p)\
                                +q1sign[a]*lamb*grid1[i1]*delta(a,ap)*delta(i1,i1p)*delta(i2,i2p)\
                                +g_list[s]*grid2[i2]*delta(i1,i1p)*delta(i2,i2p)*(1.-delta(a,ap))
                            if model=='Jahn_Teller':
                            #    H[(a*n1+i1)*n2+i2,(ap*n1+i1p)*n2+i2p]=(EJT[s]+h0JT[i1*n2+i2])*delta(a,ap)*delta(i1,i1p)*delta(i2,i2p)\
                            #    +q1sign[a]*lambdaJT[s]*qmat1[i1,i1p]*delta(a,ap)*delta(i2,i2p)\
                            #    +lambdaJT[s]*qmat2[i2,i2p]*delta(i1,i1p)*(1.-delta(a,ap))

                                HDVR[(a*n1+i1)*n2+i2,(ap*n1+i1p)*n2+i2p]=(EJT[s]+h0JT[i1*n2+i2])*delta(a,ap)*delta(i1,i1p)*delta(i2,i2p)\
                                +q1sign[a]*lambdaJT[s]*qmat1[i1,i1p]*delta(a,ap)*delta(i2,i2p)\
                                +lambdaJT[s]*qmat2[i2,i2p]*delta(i1,i1p)*(1.-delta(a,ap))

    print (count,'none-zero matrix elements out of',N*N)
# diagonalize   


  #  evals, evecs = np.linalg.eigh(H)
    evals, evecs = np.linalg.eigh(HDVR)

    A = LinearOperator((N,N), matvec=mv)
    AEa = LinearOperator((N,N), matvec=Ea_v)
    Ah01 = LinearOperator((N,N), matvec=h01_v)
    Ah02 = LinearOperator((N,N), matvec=h02_v)
    Aq1 = LinearOperator((N,N), matvec=q1_v)
    Aq2 = LinearOperator((N,N), matvec=q2_v)

    A_total=AEa+Ah01+Ah02+Aq1+Aq2

    kmax=6
    eigenvalues, eigenvectors = eigsh(A_total, k=kmax,which = 'SA')

    for i in range(kmax):
        print(evals[i],eigenvalues[i])


    delta_E=evals[1]-evals[0]
    if model=='Displaced':
        labels='D, gamma='+str(g_list[s])
    if model=='Jahn_Teller':
        labels='JT, E='+str(EJT[s])+' lambda='+str(lambdaJT[s])
    print (labels,'Theta=',delta_E/eV_per_K,' K')   

    # choose temperatures between 0.1 and 10 time the characteristic Theta=delta_E/eV_per_K
    Tmin=1.*delta_E/eV_per_K
    Tmax=3.*delta_E/eV_per_K
    Tmin=300.
    Tmax=300.
    nT=1 # number of temperature values
    deltaT=(Tmax-Tmin)/float(nT)
    T=np.zeros(nT,float)
    Probs=np.zeros((nT,N),float)
    Z=np.zeros(nT,float)
    E=np.zeros(nT,float)
    E2=np.zeros(nT,float)
    Cv=np.zeros(nT,float)
    A=np.zeros(nT,float)
    S=np.zeros(nT,float)
    for t in range(nT):
        T[t]=Tmin+t*deltaT
        T[t]=300.
    # estimators, <E>, Cv, S, and Z
        Z[t]=0.
        for i in range(N):
            Ei=(evals[i]-evals[0])/eV_per_K
            Z[t]+=np.exp(-Ei/T[t])
            E[t]+=np.exp(-Ei/T[t])*Ei
            E2[t]+=np.exp(-Ei/T[t])*Ei*Ei
        E[t]/=Z[t]
        E2[t]/=Z[t]
        Cv[t]=(E2[t]-E[t]**2)/(T[t]**2)
        A[t]=-T[t]*np.log(Z[t])
        S[t]=(E[t]-A[t])/T[t]
        for i in range(N):
            Ei=(evals[i]-evals[0])/eV_per_K
            Probs[t,i]=np.exp(-Ei/T[t])/Z[t]

    # find highest accesible index

    rho1=np.zeros((n1),float)
    rho2=np.zeros((n2),float)
    rho12=np.zeros((n1,n2),float)
    rhoa=np.zeros((na,na),float)


    for i in range(100):
        Ei=(evals[i]-evals[0])/eV_per_K
        for a in range(na):
            for ap in range(na):
                for i1 in range(n1):
                    for i2 in range(n2):
                        rhoa[a,ap]+=evecs[(a*n1+i1)*n2+i2,i]*evecs[(ap*n1+i1)*n2+i2,i]*np.exp(-Ei/T[-1])

        for i1 in range(n1):
                for i2 in range(n2):
                    for a in range(na):
                        rho1[i1]+=(evecs[(a*n1+i1)*n2+i2,i]**2)*np.exp(-Ei/T[-1])
        for i2 in range(n2):
                for i1 in range(n1):
                    for a in range(na):
                        rho2[i2]+=(evecs[(a*n1+i1)*n2+i2,i]**2)*np.exp(-Ei/T[-1])
        for i1 in range(n1):
                for i2 in range(n2):
                        for a in range(na):
                            rho12[i1,i2]+=(evecs[(a*n1+i1)*n2+i2,i]**2)*np.exp(-Ei/T[-1])
    rho1=(1./Z[-1])*rho1
    rho2=(1./Z[-1])*rho2
    rho12=(1./Z[-1])*rho12
    rhoa=(1./Z[-1])*rhoa

    print('rhoa(a,b)= ')
    print(rhoa)

    #grid1, T1    
    #convert to grid
    h1=np.zeros(n1,float)
    for i1 in range(n1):
        h1[i1]=rho1[i1]
        # multiply by gauss hermite weight
        h1[i1]*=np.exp(-grid1[i1]**2)/(np.sqrt(np.pi)*T1[0,i1]**2)
    ax_rho1.plot(grid1,h1,label=labels+' T='+str(T[-1])+' K')
    ax_rho1.legend(loc="upper right")

    h2=np.zeros(n2,float)
    for i2 in range(n2):
        h2[i2]=rho2[i2]
        # multiply by gauss hermite weight
        h2[i2]*=np.exp(-grid1[i2]**2)/(np.sqrt(np.pi)*T2[0,i2]**2)
    ax_rho2.plot(grid2,h2,label=labels+' T='+str(T[-1])+' K')
    ax_rho2.legend(loc="upper right")


    ax_EV.plot([i for i in range(N)],(evals-evals[0])/eV_per_K,label=labels)
    ax_EV.legend(loc="upper left")
    ax_E.plot(T,E,label=labels)
    ax_E.legend(loc="upper left")
    ax_Cv.plot(T,Cv,label=labels)
    ax_Cv.legend(loc="upper right")
    ax_A.plot(T,A,label=labels)
    ax_A.legend(loc="lower left")
    ax_S.plot(T,S,label=labels)
    ax_S.legend(loc="upper right")

ax_EV.set(title='E(n) vs n '+str(model),xlabel='n',ylabel='E(n)/kB (K)')
ax_E.set(title='<E> vs T '+str(model),xlabel='T (K)',ylabel='<E>/kB (K)')
ax_Cv.set(title='Cv vs T '+str(model),xlabel='T (K)',ylabel='Cv/kB ')
ax_A.set(title='A vs T '+str(model),xlabel='T (K)',ylabel='A/kB (K)')
ax_S.set(title='S vs T '+str(model),xlabel='T (K)',ylabel='S/kB')

fig_rho1.savefig('rho1_'+str(model)+'.png')
fig_rho2.savefig('rho2_'+str(model)+'.png')

fig_EV.savefig('Evsn_'+str(model)+'.png')
fig_E.savefig('EvsT_'+str(model)+'.png')
fig_Cv.savefig('CvvsT_'+str(model)+'.png')
fig_A.savefig('AvsT_'+str(model)+'.png')
fig_S.savefig('SvsT_'+str(model)+'.png')


