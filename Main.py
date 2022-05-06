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
    """
    Class to store initial conditions of a Network problem and
    store frequenly used constants for the network.

    ...

    Attributes
    ----------
    qx : numpy 3D array
        Initial condition. For i,j it contains the resource
        distribution at t=0, i.e. qij(x_n,0) for 0<n<=N[i][j]
    l : numpy matrix
        Lenghts of each edge i,j
    R : numpy matrix
        Rate of delivery outside of the network of each edge i,j
    D : numpy matrix
        Diffusion coefficient of edge i,j
    u : numpy matrix
        Average Medium velocity of edge i,j at time t
    S : numpy matrix
        Cross sectional area of edge i,j at time t
    I : vector of functions
        net rate at which resource leaves node i at time t
    N : numpy matrix
        Number of grid points along x for edge i,j
    adj : list of lists
        Adjacency matrix to efficiently tranverse the matrix.
        List entry i is a list of the indices j connected to i.
    qt : vector of functions
        Boundary condition. Resource distribution qij(0,t)=Sij*Ci(t)
        for each node i over time t
    C : (optional) vector of functions
        Not currently implemented. Concentrations functions

    Methods
    -------
    antisymmetrize(matrix, lowerleft=False)
        Makes the matrix matrix antisymmetric

    symmetrize(matrix, lowerleft=False)
        Makes the matrix matrix symmetric

    evalmatrix(matrix, t=0, printmatrix=False)
        Evaluates a matrix function at t

    evalvector(vector, t=0, printvector=False)
        Evaluates a vector function at t
    """
    def __init__(self, qx, l, R, D, u, S, I,
                 N, adj, qt, C=False):
        self.qx = qx    #Array of q at every Delta x when t=0
        self.l = l
        self.R = R
        self.D = D
        self.u = u
        self.S = S
        self.I = I
        self.N = N  #this could be optional
        self.adj = adj    #this could be optional
        self.qt = qt    #This is also optional given I
        self.C = C  #not currently implemented
        if (not C) and (not I):
            raise(ValueError("No concentration C or current I given"))
        if not I:
            self.n = len(C)
        else:
            self.n = len(I)
        self.g = u*l/(2*D)

    @staticmethod
    def antisymmetrize(matrix, lowerleft=False):
        n = len(n)
        for i in range(n):
            for j in range(i+1, n):
                if not lowerleft:
                    matrix[j][i] = -matrix[i][j]
                else:
                    matrix[i][j] = -matrix[j][i]

    @staticmethod
    def symmetrize(matrix, lowerleft=False):
        n = len(matrix)
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
    LOG2 = math.log(2)

    def __init__(self, num_roots=30):
        self.num_of_roots = num_roots
        self.x_i, self.w_i = f_tools.lag_weights_roots(self.num_of_roots)

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

    def Laplace(self, Qt, s):
        # integrate e^-t f(t/s)/s dt using gauss laguerre quadrature
        def g(t=0):
            return Qt(t/s)/s
        integral = 0
        for i in range(self.num_of_roots):
            integral = integral+self.w_i[i]*g(self.x_i[i])
        return integral

    def GSinverse(self, Qs, t, omega):
        """
        My own implementation of GS method. omega must be an even integer,
        t>0, and Qs the Laplace transform function evaluated at Omega nodes
        of the form n*ln(2)/t
        """
        #finding the kappa coefficients
        if(t == 0 or omega % 2 != 0):
            raise(ValueError("t=0 is not a valid input and omega must be even"))
        kappa = [0]*(omega+1)
        for n in range(1, omega+1):
            omega2 = omega//2
            sign = 1 if ((n+omega2) % 2 == 0) else -1
            lowK = math.floor((n+1)/2)
            highK = min(n, omega2)
            Sum = 0
            for k in range(lowK, highK+1):
                summand = k**(omega2)/(math.factorial(omega2-k)
                                       * math.factorial(k-1))
                summand = summand*(math.comb(2*k, k)*math.comb(k, n-k))
                Sum = Sum+summand
            kappa[n] = sign*Sum
        # Computing the inverse Laplace at t
        qt = 0
        for n in range(1, omega+1):
            qt = qt+kappa[n]*(Qs[n])
        qt = (self.LOG2/t)*qt
        return qt


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
        return I_lap

    def prog_matrix(self, s, t=0, return_extra=False):
        n = self.ic.n
        alpha = np.zeros([n, n])
        h = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                alpha[i][j] = math.sqrt(self.ic.u[i][j]*self.ic.u[i][j]
                                        + 4*self.ic.D[i][j]*(s+self.ic.R[i][j]))
                h[i][j] = alpha[i][j]*self.ic.l[i][j]/(2*self.ic.D[i][j])
        M = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i == j:
                    sum = 0
                    for k in range(n):
                        sum += self.ic.S[i][k]*(self.ic.u[i][k]/2 + alpha[i][k]/(2*math.tanh(h[i][j])))
                    M[i][j] = sum
                else:
                    M[i][j] = (-self.ic.S[i][j]
                               * alpha[i][j]
                               * math.exp(-self.ic.g[i][j])
                               / (2*math.sinh(h[i][j])))
        if not return_extra:
            return M
        else:
            return M, alpha, h

    def beta_matrix(self, s, t=0, return_extra=False):
        n = self.ic.n
        M, alpha, h = self.prog_matrix(s, t, True)
        b = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                b[i][j] = Solver.beta(s,
                                      self.ic.qx[i][j],
                                      self.ic.g[i][j],
                                      h[i][j],
                                      R[i][j],
                                      self.ic.u[i][j],
                                      alpha[i][j],
                                      self.ic.N[i][j])
        if not return_extra:
            return b
        else:
            return b, M, alpha, h

    def homogeneous_lap(self, x_range, t, Omega=10):
        log2 = self.ftool.LOG2
        ndim = self.ic.n
        # values of s for GS laplace inversion
        s_range = np.array([n*log2/t for n in range(1, Omega+1)])
        Ms = np.zeros([Omega, ndim, ndim])
        alphas = np.zeros([Omega, ndim, ndim])
        hs = np.zeros([Omega, ndim, ndim])
        Xs = np.zeros([Omega, ndim, ndim])
        s_ind = 0
        for s in s_range:
            # Solve the matrix system
            Ms[s_ind], alphas[s_ind], hs[s_ind] = self.prog_matrix(s, t, True)
            I = self.I_Lap_vector(s)
            C = scipy.sparse.linalg.spsolve(Ms[s_ind], I)
            # fill the X_ij(s)
            for i in range(ndim):
                for j in self.ic.adj[i]:
                    Xs[s_ind][i][j] = [C[i]*self.ic.S[i][j]]
            s_ind = s_ind + 1
        # Find qij(x,t) using stored values and GS algor

        qs = np.zeros([ndim, ndim, self.ic.N[i][j]])
        for i in range(ndim):
            for j in self.ic.adj[i]:
                x_ind = 0
                for x in x_range:
                    Qijxs = [self.Q(i, j, x, s_ind, Xs, hs) for s_ind in range(Omega)]
                    qs[i][j][x_ind] = self.ftool.GSinverse(Qijxs, t, Omega)
                    x_ind = x_ind+1




    @staticmethod
    def beta(s, qx, g, h, R, u, alpha, N):
        sum = 0
        left_term = (math.exp(h/N)-math.exp(-g/N))*(alpha-u)
        right_term = (math.exp(-h/N)-math.exp(-g/N))*(alpha+u)
        denominator = (4*(s+R)*math.sinh(h))
        for n in range(1, N+1):
            # x_n=n*l/N
            k = qx[n-1]
            coeff = k*math.exp(((1-n)/N)*g) / denominator
            sum += coeff*(math.exp(((N-n)/N)*h)*left_term
                          + math.exp(((n-N)/N)*h)*right_term)
        return sum

    def Q(self, i, j, x, s_ind, X, h):
        coeff1 = self.ic.l[i][j]-x/self.ic.l[i][j]
        coeff2 = x/self.ic.l[i][j]
        Sol = (X[s_ind][i][j]*(math.sinh(coeff1*h[s_ind][i][j])
                               / math.sinh(h[s_ind][i][j])
                               )
               * math.exp(coeff2*self.ic.g[i][j])
               + X[s_ind][j][i]*(math.sinh(coeff2*h[s_ind][i][j])
                                 / math.sinh(h[s_ind][i][j])
                                 )
               * math.exp(-coeff1*self.ic.g[i][j])
               )
        return Sol

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
            return Dm + (u[i][j] * u[i][j]) * 1 / (48 * Dm)

        D[i][j] = D_ij

    C[i] = f_tools.zerof
    for j in range(n):
        if adjacency[i][j]:
            def C_i(t=0):
                return qt(t) / S[i][j]
            C[i] = C_i
            break

# if we don't have C we need I_i(t) (net rate resource leaves i)
I = [f_tools.zerof]*n
for i in range(n):
    def I_i(t=0):
        sum = 0
        for j in range(n):
            sum = sum+u[i][j]*t/10
        return sum
    I[i] = I_i

print("u=")
InitialConditions.evalmatrix(u, printmatrix=True)
print("S=")
InitialConditions.evalmatrix(S, printmatrix=True)
print("D=")
InitialConditions.evalmatrix(D, printmatrix=True)
print("q=")
InitialConditions.evalvector(qx, printvector=True)
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
import numpy as np
x=np.zeros([5,2,2])+1
y=np.zeros([2,2])+3
z=np.array([[1,2],[4,8]])

print(x[0])
x[0]=y
print(x[0])

# %%
