# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof
# My module
from pde_network import InitialConditions
from pde_network import _f_tools
from pde_network import Solver

# ----- initial conditions setup
# parameters to ease change network input
n = 2
Ngrid = 101
Dconst = 1
uconst = 0.0
Rconst = 10.0
lconst = 1
Sconst = 0.5
Iconst = 0.3

# Networks parameters
qx = np.zeros([n, n, Ngrid]) #not used on the homogeneous solution
l = np.matrix([[0, lconst], [lconst, 0]], dtype=np.float64)
R = np.matrix([[0, Rconst], [Rconst, 0]], dtype=np.float64)
D = np.matrix([[0, Dconst], [Dconst, 0]], dtype=np.float64)
u = np.matrix([[0, uconst], [-uconst, 0]], dtype=np.float64)
S = np.matrix([[0, Sconst], [Sconst, 0]], dtype=np.float64)

# fill upper triangular part of q_ij(x,0) from 0 to l_ij
k = 0.0
for i in range(n):
    for j in range(i+1, n):
        qx[i,j] = [k for x in range(Ngrid)]
        qx[j,i] = np.flip(qx[i,j])

# We need I_i(t) (net rate resource leaves i)
I = [_f_tools.zerof]*n
def I_i(t=0):
    return Iconst
for i in range(n):
    I[i] = I_i

# ----- Time evolution
T=[0.001,0.003,0.005,0.007,0.01]
#T=[0.1,0.3,0.5,0.7,1.0]
IC_test = InitialConditions(qx, l, R, D, u, S, I)
deltax = lconst/(Ngrid-1)
x_range = np.array([k*deltax for k in range(Ngrid)])
for t in T:
    solve = Solver(IC_test)
    q = solve.homogeneous_lap(t)
    #plot the solutions
    plt.plot(x_range, q[0,1], label=f"$t = {t}$")
plt.title(f"edge (0,1) time evolution")
plt.legend()
