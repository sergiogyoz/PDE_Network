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


def populate_ij(matrix, i, j, function):
    matrix[i][j] = function


def antisymmetrize(U, lowerleft=False):
    for i in range(n):
        for j in range(i+1, n):
            if not lowerleft:
                def negUij(t=0):
                    return -U[i][j](t)
                U[j][i] = negUij
            else:
                def negUji(t=0):
                    return -U[j][i](t)
                U[i][j] = negUji


def symmetrize(S, lowerleft=False):
    for i in range(n):
        for j in range(i+1, n):
            if not lowerleft:
                S[j][i] = S[i][j]
            else:
                S[i][j] = S[j][i]


U = [[0]*n]*n
S = [[0]*n]*n
for i in range(n):
    for j in range(i, n):
        def u_ij(t=0):
            return i+j

        def S_ij(t=0):
            return 1
        populate_ij(U, i, j, u_ij)
        populate_ij(S, i, j, S_ij)

antisymmetrize(U)
symmetrize(S)


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


evalmatrix(U, printmatrix=True)
evalmatrix(S, printmatrix=True)
# %%

# %%
