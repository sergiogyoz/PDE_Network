# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof
# My module
from pde_network import InitialConditions
from pde_network import _f_tools
from pde_network import Solver

# ----- Testing non homogeneous against an exact solution
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

# fill upper triangular part of q_ij(x,0) and flip for q_ji(x,0)
deltax=math.pi/(Ngrid-1)
x_range=np.array([k*deltax for k in range(Ngrid)])
for i in range(n):
    for j in range(i+1, n):
        qx[i,j] = np.sin(x_range)+1
        qx[j,i] = np.flip(qx[i,j])

# We need I_i(t) (net rate resource leaves i)
I = [_f_tools.zerof]*n
def I_i(t):
    return -math.exp(-t)
for i in range(n):
    I[i] = I_i

# start the testing
T=[0.001, 0.01, 0.1, 1, 10]
x_range=np.array([k*deltax for k in range(Ngrid-1)])
"""T=[0.1,0.3,0.5,0.7,1.0]"""
IC_test=InitialConditions(qx, l, R, D, u, S, I)
qsum=np.zeros([2,len(T),Ngrid-1])

# solve for qhat at different times
t_ind=0
for t in T:
    solve=Solver(IC_test)
    qhat=solve.resource_y(t)
    qsum[0,t_ind]=qhat[0,1]
    #plot the solutions
    plt.plot(x_range, qhat[0,1], label=f"$t = {t}$")
    print("Int qhat at t = ", np.sum(qhat[0,1]))
    t_ind+=1
plt.title(f"$\int \hat q$ edge (0,1) time evolution")
plt.legend()
plt.show()

# Solve for q tilde at different times
plt.figure()
t_ind=0
for t in T:
    solve=Solver(IC_test)
    qtilde=solve.resource_z(t)
    qsum[1,t_ind]=qtilde[0,1]
    #plot the solutions
    plt.plot(x_range, qtilde[0,1], label=f"$t = {t}$")
    print("Int qtilde at t = ", np.sum(qtilde[0,1]))
    t_ind+=1
plt.title(f"$\int qtil$ edge (0,1) time evolution")
plt.legend()
plt.show()

# Plot the solution against the exact solution
plt.figure()
t_ind=0
for t in T:
    #plot the solutions
    plt.plot(x_range,
             qsum[0,t_ind]+qsum[1,t_ind],
             label=f"$t = {t}$")
    print("Int qh+qt at t = ", np.sum(qsum[0,t_ind]+qsum[1,t_ind]))
    t_ind+=1
plt.title(f"$\int q=\int qtil+\int \hat q$ edge (0,1) time evolution")
plt.legend()
plt.show()

# Plot exact solutions to compare
for t in T:
    #plot the solutions
    plt.plot(x_range,
             math.exp(-t)*np.sin(x_range)+1,
             label=f"$t = {t}$")
    print("exact Int q at t = ", 2*math.exp(-t)+math.pi)
plt.title(f"exact solution edge (0,1) time evolution")
plt.legend()
plt.show()

# %%
