from mpmath import *
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 15; mp.pretty = True
tt = [0.001, 0.01, 0.1, 1, 10]
fs = lambda s: 1/(s+1)**2
ft = lambda t: t*exp(-t)

f=[]
t=np.arange(0,10,0.1)
for time in t:
    f.append( ft(time) )

plt.plot(t,f);

print(ft(tt[0]))
print(invertlaplace(fp,tt[0],method='talbot'))

