# %%
import math
import numpy as np
import matplotlib.pyplot as plt
from sys import getsizeof

# Laplace transform uses Gauss–Laguerre quadrature
from sympy import poly, laguerre
from sympy.abc import x

# sparse linear solver
import scipy.sparse
import scipy.sparse.linalg


# %%
class InitialConditions:
    def __init__(self, l, R, D, u, S, qx,
                 C=False, I=False, qt=False, adjcency=False):
        self.qx = qx
        self.l = l
        self.R = R
        self.D = D
        self.u = u
        self.S = S
        self.C = C
        self.I = I
        self.qt = qt
        if (not C) and (not I):
            raise(ValueError("No concentration C or current I given"))
        if not I:
            self.n = len(C)
        else:
            self.n = len(I)
        self.adj = adjacency

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


class f_tools:

    def __init__(self,num_roots=30):
        self.num_of_roots = num_roots
        self.x_i, self.w_i = f_tools.lag_weights_roots(
            self.num_of_roots)

    @staticmethod
    def zerof(t=0):
        return 0

    @staticmethod
    def onef(t=0):
        return 1

    @staticmethod
    def consf(constant):
        def cons(t=0):
            return constant
        return cons

    @staticmethod
    def lag_weights_roots(n):
        roots = poly(laguerre(n, x)).all_roots()
        x_i = [rt.evalf(20) for rt in roots]
        w_i = [(rt / ((n + 1)
                * laguerre(n + 1, rt)) ** 2).evalf(20) for rt in roots]
        return x_i, w_i

    def Laplace(self, func, s):
        # integrate e^-t f(t/s)/s dt using gauss laguerre quadrature
        def g(t=0):
            return func(t/s)/s
        integral = 0
        for i in range(self.num_of_roots):
            integral = integral+self.w_i*g(self.x_i)
        return integral


class Solver:
    ic = InitialConditions()
    ftool = f_tools()

    def __init__(self, initial_conditions):
        self.ic = initial_conditions
        self.ftool = f_tools()

    def I_Lap_vector(self, s):
        n = self.ic.n
        I_lap = np.zeros(n)
        for i in range(n):
            I_lap[i] = self.ftool.Laplace(self.ic.I[i], s)

    def prog_matrix(self, s, t=0, return_extra=False):
        n = self.ic.n
        alpha = np.zeros([n, n])
        h = np.zeros([n, n])
        g = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                alpha[i][j] = math.sqrt(self.ic.u[i][j](t)*self.ic.u[i][j](t)
                                        + 4*self.ic.D[i][j](t)*(s+self.ic.R[i][j]))
                g[i][j] = self.ic.u[i][j](t)*self.ic.l[i][j]/(2*self.ic.D[i][j](t))
                h[i][j] = alpha[i][j]*self.ic.l[i][j]/(2*self.ic.D[i][j](t))
        M = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i == j:
                    sum = 0
                    for k in range(n):
                        sum += self.ic.S[i][k](t)*(self.ic.u[i][k](t)/2 + alpha[i][k]/(2*math.tanh(h[i][j])))
                    M[i][j] = sum
                else:
                    M[i][j] = (-self.ic.S[i][j](t)
                               * alpha[i][j]
                               * math.exp(-g[i][j])
                               / (2*math.sinh(h[i][j])))
        if not return_extra:
            return M
        else:
            return M, alpha, h, g

    def beta_matrix(self, N, s, t=0, return_extra=False):
        n = self.ic.n
        M, alpha, h, g = self.prog_matrix(s, t, return_extra)
        b = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                b[i][j] = Solver.beta(s,
                                      self.ic.qx[i][j],
                                      g[i][j],
                                      h[i][j],
                                      R[i][j],
                                      self.ic.u[i][j](t),
                                      self.ic.l[i][j],
                                      alpha[i][j],
                                      N)
        if not return_extra:
            return b
        else:
            return b, M, alpha, h, g

    @staticmethod
    def beta(s, qx, g, h, R, u, l, alpha, N):
        sum = 0
        left_term = (math.exp(h/N)-math.exp(-g/N))*(alpha-u)
        right_term = (math.exp(-h/N)-math.exp(-g/N))*(alpha+u)
        denominator = (4*(s+R)*math.sinh(h))
        for n in range(1, N+1):
            k = qx((n-1)*l/N)
            coeff = k*math.exp(((1-n)/N)*g) / denominator
            sum += coeff*(math.exp(((N-n)/N)*h)*left_term
                          + math.exp(((n-N)/N)*h)*right_term)
        return sum


    C = scipy.sparse.linalg.spsolve(M, I)


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

# fill q_ij(x,0) from 0 to l_ij
k=0.01
qx=[[0]*n]*n
for i in range(n):
    for j in range(i+1,n):
        def g(x):
            if (0<=x and x<=l[i][j]):
                return k
            else:
                return 0
        qx[i][j]=g
        
# fill q_ij(0,t)
qt = [f_tools.zerof]*n
for i in range(n):
    def q_i(t=0):
        return t/10
    qt[i] = q_i


# fill D and C
for i in range(n):
    for j in range(n):
        def D_ij(t=0):
            return Dm + (u[i][j](t) * u[i][j](t)) * 1 / (48 * Dm)

        D[i][j] = D_ij

    C[i] = f_tools.zerof
    for j in range(n):
        if adjacency[i][j]:
            def C_i(t=0):
                return qt(t) / S[i][j](t)
            C[i] = C_i
            break

# if we don't have C we need I_i(t) (net rate resource leaves i)
I = [f_tools.zerof]*n
for i in range(n):
    def I_i(t=0):
        sum = 0
        for j in range(n):
            sum = sum+u[i][j](t)*t/10
        return sum
    I[i] = I_i

print("u=")
InitialConditions.evalmatrix(u, printmatrix=True)
print("S=")
InitialConditions.evalmatrix(S, printmatrix=True)
print("D=")
InitialConditions.evalmatrix(D, printmatrix=True)
print("q=")
InitialConditions.evalvector(q, printvector=True)
print("I=")
InitialConditions.evalvector(I, printvector=True)

# %%
"""A=np.zeros([2,2])
A[1][1]=1
A[0][0]=2
b=np.zeros(2)
b[1]=5
b[0]=3.215
x=scipy.sparse.linalg.spsolve(A,b)
print(x)
print(A.dot(x)-b)"""
# %%
