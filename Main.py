# %%
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof


class InitialConditions:
    def __init__(self, l, R, D, u, S, C=False, I=False):
        self.l = l
        self.R = R
        self.D = D
        self.u = u
        self.S = S
        self.C = C
        self.I = I

    @staticmethod
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

    @staticmethod
    def symmetrize(matrix, lowerleft=False):
        for i in range(n):
            for j in range(i+1, n):
                if not lowerleft:
                    matrix[j][i] = matrix[i][j]
                else:
                    matrix[i][j] = matrix[j][i]

    # useful functions for matrix and vector evaluation
    @staticmethod
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

    @staticmethod
    def evalvector(vector, t=0, printvector=False):
        n = len(vector)
        vec = np.zeros(n)
        for i in range(n):
            vec[i] = vector[i](t)
        if printvector:
            print(vec)
        return vec

    @staticmethod
    def zerof(t=0):
        return 0

    @staticmethod
    def onef(t=0):
        return 1

    @staticmethod
    def consf(constant):
        def cons(t=0):
            return constant;
        return cons;

# %%
# initial conditions setup
n = 2
# We start by saving all the constant coefficients of the network
l = np.matrix([[0, 1], [1, 0]])
R = np.matrix([[0, 0.1], [0.1, 0]])
Dm = 0.1


# zero, one, and constant functions



adjacency = [[0, 1], [-1, 0]]  # adjacency matrix
u = [[0]*n]*n  # velocities
S = [[0]*n]*n  # cross sectional areas
D = [[0]*n]*n  # diffusion outside of the network
C = [0]*n  # concentration at the node
# fill U and S
for i in range(n):
    for j in range(i, n):
        def u_ij(t=0):
            return i+j

        def S_ij(t=0):
            return 1/2

        u[i][j] = u_ij
        S[i][j] = S_ij

InitialConditions.antisymmetrize(u)
InitialConditions.symmetrize(S)

# fill q_i(0,t)
q = [0]*n
for i in range(n):
    def q_i(t=0):
        return t/10
    q[i] = q_i


# fill D and C
for i in range(n):
    for j in range(n):
        def D_ij(t=0):
            return Dm + (u[i][j](t) ** 2) * 1 / (48 * Dm)

        D[i][j] = D_ij

    C[i] = zerof
    for j in range(n):
        if adjacency[i][j]:
            def C_i(t=0):
                return q(t) / S[i][j](t)
            C[i] = C_i
            break

# if we don't have C we need I_i(t) (net rate resource leaves i)
I = [zerof]*n
for i in range(n):
    def I_i(t=0):
        sum = 0
        for j in range(n):
            sum = sum+u[i][j](t)*t/10
        return sum
    I[i] = I_i







print("u=")
evalmatrix(u, printmatrix=True)
print("S=")
evalmatrix(S, printmatrix=True)
print("D=")
evalmatrix(D, printmatrix=True)
print("q=")
evalvector(q, printvector=True)
print("I=")
evalvector(I, printvector=True)


# %%
def one(x=0):
    return 1
def two(x=0):
    return 2

x=[one]
print(x[0]())
x[0]=two
print(x[0]())
print(one())

# %%
