from mpmath import *

mp.dps = 15; mp.pretty = True
tt = [0.001, 0.01, 0.1, 1, 10]
fp = lambda p: 1/(p+1)**2
ft = lambda t: t*exp(-t)
ft(tt[0]),ft(tt[0])-invertlaplace(fp,tt[0],method='talbot')
(0.000999000499833375, 8.57923043561212e-20)