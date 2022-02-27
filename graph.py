import matplotlib.pyplot as plt
import numpy as np
import sys
x1x2file=sys.argv[1]
h1file=sys.argv[2]
h2file=sys.argv[3]
data = np.loadtxt(x1x2file)
datah1=np.loadtxt(h1file)
datah2=np.loadtxt(h2file)
x1 = data[:,1]
x2 = data[:,2]
q1=datah1[:,0]
h1=datah1[:,1]
q2=datah2[:,0]
h2=datah2[:,1]

# the histogram of the data
n1, bins, patches = plt.hist(x1, 50, density=True, facecolor='g', alpha=0.5)
n2, bins, patches = plt.hist(x2, 50, density=True, facecolor='r', alpha=0.5)
plt.plot(q1,h1)
plt.plot(q2,h2)

plt.xlabel('q')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('f.png')
