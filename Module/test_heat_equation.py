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
n = 2
Ngrid = 101
Dconst = 1
uconst = 0.0
Rconst = 0.0
lconst = math.pi
Sconst = 1
# Networks parameters
qx = np.zeros([n, n, Ngrid])
l = np.matrix([[0, lconst], [lconst, 0]], dtype=np.float64)
R = np.matrix([[0, Rconst], [Rconst, 0]], dtype=np.float64)
D = np.matrix([[0, Dconst], [Dconst, 0]], dtype=np.float64)
u = np.matrix([[0, uconst], [-uconst, 0]], dtype=np.float64)
S = np.matrix([[0, Sconst], [Sconst, 0]], dtype=np.float64)
# Fill upper triangular part of q_ij(x,0) and flip for q_ji(x,0)
deltax=math.pi/(Ngrid-1)
x_range=np.array([k*deltax for k in range(Ngrid)])
for i in range(n):
    for j in range(i+1, n):
        qx[i,j] = np.sin(x_range)+1
        qx[j,i] = np.flip(qx[i,j])

# We need I_i(t) (net rate resource that leaves i)
I = [_f_tools.zerof]*n
def I_i(t):
    return -math.exp(-t)
for i in range(n):
    I[i] = I_i

# start the testing
T=[0.001, 0.01, 0.1, 1, 10]
IC_test = InitialConditions(qx, l, R, D, u, S, I)
qsum = np.zeros([2, len(T), Ngrid])
figsize=(6,5); dpi=150
# solve for qhat at different times
plt.figure(figsize=figsize,dpi=dpi)
t_ind = 0
for t in T:
    solve = Solver(IC_test)
    qhat = solve.non_homogeneous_lap(t)
    qsum[0,t_ind] = qhat[0,1]
    t_ind += 1
    # plot the solutions
    plt.plot(x_range, qhat[0,1], label=f"$t = {t}$")
plt.title(f" $qhat$ time evolution")
plt.legend()
plt.show()

# Solve for q tilde at different times
plt.figure(figsize=figsize,dpi=dpi)
t_ind = 0
for t in T:
    solve = Solver(IC_test)
    qtilde = solve.q_tilde(t)
    qsum[1,t_ind] = qtilde[0,1]
    t_ind += 1
    # plot the solutions
    plt.plot(x_range, qtilde[0,1], label=f"$t = {t}$")
plt.title(f"$qtilde$ time evolution")
plt.legend()
plt.show()
# Plot the solution against the exact solution
plt.figure(figsize=figsize,dpi=dpi)
t_ind = 0
for t in T:
    # plot the solutions
    plt.plot(x_range,
             qsum[0,t_ind]+qsum[1,t_ind],
             label=f"$t = {t}$")
    t_ind += 1
plt.title(f"q=qtilde+qhat edge (0,1) time evolution")
plt.legend()
plt.show()

# Plot exact solutions to compare
plt.figure(figsize=figsize,dpi=dpi)
for t in T:
    # plot the solutions
    plt.plot(x_range,
             math.exp(-t)*np.sin(x_range)+1,
             label=f"$t = {t}$")
plt.title(f"exact solution edge (0,1) time evolution")
plt.legend()
plt.show()

# Plot exact solutions to compare
plt.figure(figsize=figsize,dpi=dpi)
t_ind = 0
for t in T:
    # plot the solutions
    plt.plot(x_range,
             (qsum[0,t_ind]+qsum[1,t_ind]
             -(math.exp(-t)*np.sin(x_range)+1))
             /(math.exp(-t)*np.sin(x_range)+1),
             label=f"$t = {t}$")
    t_ind += 1
plt.title(f"Relative error at different times")
plt.legend()
plt.show()

# %%
