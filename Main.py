# %%
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof
# %%
# initial conditions setup
n = 2
# We start by saving all the constant coefficients of the network
l_ij = np.matrix([[0, 1], [1, 0]])
R_ij = np.matrix([[0, 0.1], [0.1, 0]])
Dm = 0.1


# zero function
def zerof(t=0):
    return 0


# make a matrix symmetric or anti-symmetric
def antisymmetrize(matrix, lowerleft=False):
    for i in range(n):
        for j in range(i+1, n):
            if not lowerleft:
                def negUij(t=0):
                    return -matrix[i][j](t)
                matrix[j][i] = negUij
            else:
                def negUji(t=0):
                    return -matrix[j][i](t)
                matrix[i][j] = negUji


def symmetrize(matrix, lowerleft=False):
    for i in range(n):
        for j in range(i+1, n):
            if not lowerleft:
                matrix[j][i] = matrix[i][j]
            else:
                matrix[i][j] = matrix[j][i]


conmatrix = [[0, 1], [-1, 0]]  # connectivity matrix
U = [[0]*n]*n  # velocities
S = [[0]*n]*n  # cross sectional areas
D = [[0]*n]*n  # diffusion outside of the network
C = [0]*n  # concentration at the node
# fill U and S
for i in range(n):
    for j in range(i, n):
        def u_ij(t=0):
            return i+j

        def S_ij(t=0):
            return 1

        U[i][j] = u_ij
        S[i][j] = S_ij

antisymmetrize(U)
symmetrize(S)

# fill D and C
for i in range(n):
    for j in range(n):
        def D_ij(t=0):
            return Dm + (U[i][j](t) ** 2) * 1 / (48 * Dm)

        def C_i(qi,t=0):
            if conmatrix[i][j]:
                return qi / S[i][j](t)
            else:
                return zerof


# useful functions for matrix and vector evaluation
def evalmatrix(matrix, t=0, printmatrix=False):
    n = len(matrix)
    m = len(matrix[0])
    mat = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            mat[i, j] = matrix[i][j](t)
    if printmatrix:
        print(mat)
    return mat


def evalvector(vector, t=0, printvector=False):
    n = len(vector)
    vec = np.zeros(n)
    for i in range(n):
        vec[i] = vector[i](t)
    if printvector:
        print(vec)
    return vec


evalmatrix(U, printmatrix=True)
evalmatrix(S, printmatrix=True)



# %%
def add2(x):
    return x + 2

print(1 + (add2(3)) ** 2)
# %%
