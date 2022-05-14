import math
import numpy as np
# Laplace transform uses Gauss–Laguerre quadrature
from sympy import poly, laguerre
from sympy.abc import x
# sparse linear solver
import scipy.sparse.linalg


# %%
class InitialConditions:
    """
    Class to store initial conditions of a Network problem and
    store frequenly used constants for the network. Make sure
    to use np.arrays with specified dtype

    ...

    Attributes
    ----------
    qx : numpy 3D array
        Initial condition. For i,j it contains the resource
        distribution at t=0, i.e. qij(x_n,0) for 0<n<=N[i,j]
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
    qt : (not implemented, optional) vector of functions
        Boundary condition. Resource distribution qij(0,t)=Sij*Ci(t)
        for each node i over time t
    C : (not implemented, optional) vector of functions
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
    def __init__(self, qx=np.zeros(0), l=np.zeros(0), R=np.zeros(0),
                 D=np.zeros(0), u=np.zeros(0), S=np.zeros(0), I=np.array([1]),
                 qt=False, C=False):
        self.qx = qx    # 3D Array of qij at every Delta x at current t
        self.l = l
        self.R = R
        self.D = D
        self.u = u
        self.S = S
        self.I = I
        self.n = len(l)

        self.N = np.zeros([self.n, self.n], dtype=np.int32)
        self.adj = [ [] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                self.N[i,j] = len(qx[i,j])
                if l[i,j]>0:
                    self.adj[i].append(j)
        self.g = np.divide(u*l, 2*D, out=np.zeros_like(u), where=(D != 0))

        self.qt = qt    # Not currently implemented. Optional
        self.C = C  # Not currently implemented
        if (not C) and (not I):
            raise(ValueError("No concentration C or current I given"))

    def print_IC(self, t=0.0, stop_at=3, show_extra=False):
        print("Number of nodes= ", self.n)
        extra_text=""
        if self.n>stop_at:
            extra_text=" ... "
        print("l= ")
        print( self.l[:stop_at, :stop_at], extra_text)
        print("R= ")
        print( self.R[:stop_at, :stop_at], extra_text)
        print("D= ")
        print( self.D[:stop_at, :stop_at], extra_text)
        print("u= ")
        print( self.u[:stop_at, :stop_at], extra_text)
        print("S= ")
        print( self.S[:stop_at, :stop_at], extra_text)
        print("I= ")
        I_mat=InitialConditions.evalvector(self.I, t)
        print( I_mat[:stop_at], extra_text)
        if show_extra:
            print("N= ")
            print( self.N[:stop_at, :stop_at], extra_text)
            print("Adjacency list= ")
            print( self.adj[:stop_at], extra_text)

    @staticmethod
    def antisymmetrize(matrix, lowerleft=False):
        n = len(matrix)
        for i in range(n):
            for j in range(i+1, n):
                if not lowerleft:
                    matrix[j,i] = -matrix[i,j]
                else:
                    matrix[i,j] = -matrix[j,i]

    @staticmethod
    def symmetrize(matrix, lowerleft=False):
        n = len(matrix)
        for i in range(n):
            for j in range(i+1, n):
                if not lowerleft:
                    matrix[j,i] = matrix[i,j]
                else:
                    matrix[i,j] = matrix[j,i]

    def fill_qx(self, lowerleft=False):
        n = self.n
        for i in range(n):
            for j in range(i+1,n):
                for k in range(self.N[i,j]):
                    self.qx[j,i,self.N[i,j]-1-k] = self.qx[i,j,k]

    # useful functions for matrix and vector evaluation
    @staticmethod
    def evalmatrix(matrix_func, t=0.0, printmatrix=False):
        n = len(matrix_func)
        m = len(matrix_func[0])
        mat = np.zeros([n, m])
        for i in range(n):
            for j in range(m):
                mat[i, j] = matrix_func[i][j](t)
        if printmatrix:
            print(mat)
        return mat

    @staticmethod
    def evalvector(vector, t=0.0, printvector=False):
        n = len(vector)
        vec = np.zeros(n)
        for i in range(n):
            vec[i] = vector[i](t)
        if printvector:
            print(vec)
        return vec


class _f_tools:
    LOG2 = math.log(2)

    def __init__(self, num_roots=30):
        self.num_of_roots = num_roots
        self.x_i, self.w_i = _f_tools.lag_weights_roots(self.num_of_roots)

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
            qt = qt+kappa[n]*(Qs[n-1])
        qt = (self.LOG2/t)*qt
        return qt

    def zerof(t=0):
        return 0


class Solver:
    """
    A solver takes initial conditions and finds qhat and q tilde
    """
    ic = InitialConditions()
    ftool = _f_tools()
    def __init__(self, initial_conditions):
        self.ic = initial_conditions
        self.ftool = _f_tools()

    def I_Lap_vector(self, s):  # verified
        n = self.ic.n
        I_lap = np.zeros(n)
        for i in range(n):
            I_lap[i] = self.ftool.Laplace(self.ic.I[i], s)
        return I_lap

    def prog_matrix(self, s, return_extra=False):  # verified
        n = self.ic.n
        alpha = np.zeros([n, n])
        h = np.zeros([n, n])
        for i in range(n):
            for j in self.ic.adj[i]:
                alpha[i,j] = math.sqrt(self.ic.u[i,j]*self.ic.u[i,j]
                                        + 4*self.ic.D[i,j]*(s+self.ic.R[i,j]))
                h[i,j] = alpha[i,j]*self.ic.l[i,j]/(2*self.ic.D[i,j])
        M = np.zeros([n, n])
        for i in range(n):
            # diagonal elements
            sum = 0
            for k in self.ic.adj[i]:
                sum += self.ic.S[i,k]*(self.ic.u[i,k]/2 + alpha[i,k]/(2*math.tanh(h[i,k])))
            M[i][i] = sum
            # non diagonal elements
            for j in self.ic.adj[i]:
                M[i][j] = (-self.ic.S[i,j]
                            * alpha[i,j]
                            * math.exp(-self.ic.g[i,j])
                            / (2*math.sinh(h[i,j])))
        if not return_extra:
            return M
        else:
            return M, alpha, h

    def beta_matrix(self, s, return_extra=False):
        n = self.ic.n
        M, alpha, h = self.prog_matrix(s, True)
        b = np.zeros([n, n])
        for i in range(n):
            for j in self.ic.adj[i]:
                b[i][j] = Solver._beta(s,
                                      self.ic.qx[i,j],
                                      self.ic.g[i,j],
                                      h[i,j],
                                      self.ic.R[i,j],
                                      self.ic.u[i,j],
                                      alpha[i,j],
                                      self.ic.N[i,j]-1)
        if not return_extra:
            return b
        else:
            return b, M, alpha, h

    def homogeneous_lap(self, t, Omega=10):
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
            Ms[s_ind], alphas[s_ind], hs[s_ind] = self.prog_matrix(s, True)
            I = self.I_Lap_vector(s)
            C = scipy.sparse.linalg.spsolve(Ms[s_ind], I)
            # fill the X_ij(s)
            for i in range(ndim):
                for j in self.ic.adj[i]:
                    Xs[s_ind,i,j] = C[i]*self.ic.S[i,j]
            s_ind = s_ind + 1
        # Find qij(x,t) using stored values and GS algor
        qt = np.zeros([ndim, ndim, self.ic.N[i,j]])
        for i in range(ndim):
            for j in self.ic.adj[i]:
                x = 0.0
                dx = self.ic.l[i,j]/(self.ic.N[i,j]-1)
                for x_ind in range(self.ic.N[i,j]):
                    Qijxs = [self._Q_homo(i, j, x, s_ind, Xs, hs) for s_ind in range(Omega)]
                    qt[i,j,x_ind] = self.ftool.GSinverse(Qijxs, t, Omega)
                    x = x+dx
        return qt

    def non_homogeneous_lap(self, t, Omega=10):
        log2 = self.ftool.LOG2
        ndim = self.ic.n
        # values of s for GS laplace inversion
        s_range = np.array([n*log2/t for n in range(1, Omega+1)])
        Ms = np.zeros([Omega, ndim, ndim])
        alphas = np.zeros([Omega, ndim, ndim])
        hs = np.zeros([Omega, ndim, ndim])
        Xs = np.zeros([Omega, ndim, ndim])
        betas = np.zeros([Omega, ndim, ndim])
        s_ind = 0
        for s in s_range:
            # fill out the betas, propagation matrix, alpha, and h
            betas[s_ind], Ms[s_ind], alphas[s_ind], hs[s_ind] = (
                self.beta_matrix(s, True))
            I = self.I_Lap_vector(s)
            beta_i = np.sum(betas[s_ind], axis=1)
            # Solve the matrix system
            C = scipy.sparse.linalg.spsolve(Ms[s_ind], I+beta_i)
            # fill the X_ij(s)
            for i in range(ndim):
                for j in self.ic.adj[i]:
                    Xs[s_ind,i,j] = C[i]*self.ic.S[i,j]
            s_ind = s_ind + 1

        # Find qij(x,t) using stored values and GS algor
        qt = np.zeros([ndim, ndim, self.ic.N[i,j]])
        for i in range(ndim):
            for j in self.ic.adj[i]:
                x = 0.0
                dx = self.ic.l[i,j]/(self.ic.N[i,j]-1)
                # It only goes up to N-1
                for x_ind in range(self.ic.N[i,j]):
                    Qijxs = [0.0]*Omega
                    for s_ind in range(Omega):
                        Qijxs[s_ind] = self._Q_homo(i, j, x, s_ind, Xs, hs)
                    qt[i,j,x_ind] = self.ftool.GSinverse(Qijxs, t, Omega)
                    x = x+dx
        return qt

    def total_resource_y(self, t, Omega=10):
        log2 = self.ftool.LOG2
        ndim = self.ic.n
        # values of s for GS laplace inversion
        s_range = np.array([n*log2/t for n in range(1, Omega+1)])
        Ms = np.zeros([Omega, ndim, ndim])
        alphas = np.zeros([Omega, ndim, ndim])
        hs = np.zeros([Omega, ndim, ndim])
        Xs = np.zeros([Omega, ndim, ndim])
        betas = np.zeros([Omega, ndim, ndim])
        s_ind = 0
        for s in s_range:
            # fill out the betas, propagation matrix, alpha, and h
            betas[s_ind], Ms[s_ind], alphas[s_ind], hs[s_ind] = (
                self.beta_matrix(s, True))
            I = self.I_Lap_vector(s)
            beta_i=np.sum(betas[s_ind], axis=1)
            # Solve the matrix system
            C = scipy.sparse.linalg.spsolve(Ms[s_ind], I+beta_i)
            # fill the X_ij(s)
            for i in range(ndim):
                for j in self.ic.adj[i]:
                    Xs[s_ind,i,j] = C[i]*self.ic.S[i,j]
            s_ind = s_ind + 1

        # Find total resource \Int_0^l qij(x,t) using stored values and GS algor
        Tq = np.zeros([ndim, ndim, self.ic.N[i,j]-1])
        for i in range(ndim):
            for j in self.ic.adj[i]:
                # It goes from 1 to up to N (number of intervals) so N=Nij-1
                for n in range(1, self.ic.N[i,j]):
                    Yijxs = [0.0]*Omega
                    for s_ind in range(Omega):
                        Yijxs[s_ind] = self._Y(n,
                                               s_range[s_ind],
                                               self.ic.g[i,j],
                                               hs[s_ind,i,j],
                                               self.ic.R[i,j],
                                               self.ic.u[i,j],
                                               alphas[s_ind,i,j],
                                               self.ic.N[i,j]-1,
                                               self.ic.l[i,j],
                                               Xs[s_ind,i,j],
                                               Xs[s_ind,j,i])
                    Tq[i,j,n-1] = self.ftool.GSinverse(Yijxs, t, Omega)
        return Tq



    def q_tilde(self, t):
        qt = np.zeros_like(self.ic.qx)
        n = self.ic.n
        for i in range(n):
            for j in self.ic.adj[i]:
                lambdas, As, M= self._bound_parameters(i, j, t)
                ms = np.array([m for m in range(1,M+1)])
                N = self.ic.N[i,j]-1
                dx = self.ic.l[i,j]/N  # x=(n-1)*dx= x_ind*dx
                for x_ind in range(N+1):
                    sines = np.sin(ms*math.pi*x_ind*dx/ self.ic.l[i,j])
                    qt[i,j,x_ind]= np.sum(As*np.exp(lambdas*t)*sines)
        return qt

    @staticmethod
    def _beta(s, qx, g, h, R, u, alpha, N): # re-checked
        """N is the number of INTERVALS, not grid points"""
        sum = 0
        left_term = (math.exp(h/N)-math.exp(-g/N))*(alpha-u)
        right_term = (math.exp(-h/N)-math.exp(-g/N))*(alpha+u)
        denominator = (4*(s+R)*math.sinh(h))
        for n in range(1, N+1):
            # x_n=(n-1)*l/N since index of qx starts at 0
            k = qx[n-1]
            coeff = k*math.exp(((1-n)/N)*g) / denominator
            sum += coeff*(math.exp(((N-n)/N)*h)*left_term
                          + math.exp(((n-N)/N)*h)*right_term)
        return sum

    @staticmethod
    def _f(n, k, g, h, D, u, alpha, N):  # verified
        """
        k= qij(x_ind)
        N is the number of intervals, not grid points.
        Using the convention k if l*(n-1)/N<=x<l*n/N
        """
        if n < 1:
            raise ValueError(f"n={n} must be >=1")
        leftcoeff = (2*D*k*math.exp(g+h))/(alpha*(u+alpha))
        rightcoeff = (2*D*k*math.exp(g+h))/(alpha*(u-alpha))
        left_term = math.exp(-(n/N)*(g+h))-math.exp(-((n-1)/N)*(g+h))
        right_term = math.exp((n/N)*(h-g))-math.exp(((n-1)/N)*(h-g))
        Sol = leftcoeff*left_term - rightcoeff*right_term
        return Sol

    def _lambda_ij(self, m, i, j):
        L = -(m*m*self.ic.D[i,j]*math.pi*math.pi
                /(self.ic.l[i,j]*self.ic.l[i,j])
              + self.ic.u[i,j]*self.ic.u[i,j]
                /(4*self.ic.D[i,j])
              + self.ic.R[i,j]
              )
        return L
    
    def _mu_ij(self, m, i, j):
        M = (8*self.ic.D[i,j]*self.ic.D[i,j]*math.pi*m
             / (self.ic.u[i,j]*self.ic.u[i,j]*self.ic.l[i,j]*self.ic.l[i,j]
                + 4*self.ic.D[i,j]*self.ic.D[i,j]*math.pi*math.pi*m*m
                )
             )
        return M

    def _A(self, m, i, j):
        N = self.ic.N[i,j]-1
        mu = self._mu_ij(m, i, j)
        sign=1 if m%2==0 else -1
        term1 = mu*(self.ic.qx[i,j,0]
                    - sign*self.ic.qx[i,j,N-1]*math.exp(-self.ic.g[i,j]))
        sum = 0
        for n in range(N):
            expg = math.exp((-n/N)*self.ic.g[i,j] )
            diff = self.ic.qx[i,j,n]-self.ic.qx[i,j,n-1]
            trig = ((self.ic.g[i,j]/(math.pi*m))
                    * math.sin(m*n*math.pi/N) + math.cos(m*n*math.pi/N)
                    )
            sum += expg*diff*trig
        return term1 + mu*sum

    def _bound_parameters(self, i, j, t, eps=10**(-4.5)):
        """
        Returns A_m and lambda_m f and M so that the relative 
        error is less than eps
        """
        lambdas = []
        As = []
        M = 1
        sum = 0
        c = 2*math.pi*math.pi*self.ic.D[i,j]*t/(self.ic.l[i,j]*self.ic.l[i,j])
        # first term
        lam = self._lambda_ij(M, i, j)
        # check condition
        while True:
            lambdas.append(lam)
            e2lambt = math.exp(lam*t)
            sum += e2lambt
            if (e2lambt < eps*M*c*sum):
                break
            if M >= 100:
                print("did not converge, stop at M=50")
                break
            #if not then increase M and try again
            M += 1
            lam = self._lambda_ij(M, i, j)
        # calculate and store the coefficients
        lambdas = np.array(lambdas)
        As = np.array([ self._A(m, i, j) for m in range(1, M+1) ])
        return lambdas, As, M

    def _Q_homo(self, i, j, x, s_ind, X, h):
        coeff1 = (self.ic.l[i,j]-x)/self.ic.l[i,j]
        coeff2 = x/self.ic.l[i,j]
        Sol = (X[s_ind,i,j]*(math.sinh(coeff1*h[s_ind,i,j])
                             / math.sinh(h[s_ind,i,j])
                             )
               * math.exp(coeff2*self.ic.g[i,j])
               + X[s_ind,j,i]*(math.sinh(coeff2*h[s_ind,i,j])
                               / math.sinh(h[s_ind,i,j])
                               )
               * math.exp(-coeff1*self.ic.g[i,j])
               )
        return Sol

    # Not working, not really used but it would be good for verification
    def _Q_non_homo(self, i, j, x, x_ind, s_ind, alpha, beta, X, h):
        """
        if n = x_ind+1 then x=l*(n-1)/N
        """
        n = x_ind+1
        f_ijxs=Solver._f(n, 
                        self.ic.qx[i, j, x_ind],
                        self.ic.g[i,j],
                        h[s_ind,i,j],
                        self.ic.D[i,j],
                        self.ic.u[i,j],
                        alpha[s_ind,i,j],
                        self.ic.N[i,j]-1)
        A = (beta[s_ind,i,j]/alpha[s_ind,i,j]
            + (X[s_ind,j,i]*math.exp(-self.ic.g[i,j])
               - X[s_ind,i,j]*math.exp(-h[s_ind,i,j])
               )
               / (2*math.sinh(h[s_ind,i,j])) 
            )
        B = (-beta[s_ind,i,j]/alpha[s_ind,i,j]
             + (X[s_ind,i,j]*math.exp(h[s_ind,i,j])
                - X[s_ind,j,i]*math.exp(-self.ic.g[i,j])
                )
                / (2*math.sinh(h[s_ind,i,j]))
            )
        expA = math.exp((self.ic.g[i,j]+h[s_ind,i,j])*(x/self.ic.l[i,j]))
        expB = math.exp((self.ic.g[i,j]-h[s_ind,i,j])*(x/self.ic.l[i,j]))
        Sol = A*expA+B*expB+f_ijxs
        return Sol

    @staticmethod
    def _Y(n, s, g, h, R, u, alpha, N, l, Xij, Xji):
        """N is the number of INTERVALS, not grid points"""
        # I'll make a term for eta and each X term and then add and multiply together
        eta = (N*math.exp(h))/(4*(s+R)*l*math.sinh(h))
        Xij1 = Xij*(math.exp(((n-1) / N) * (g - h))
                    - math.exp((n / N) * (g - h)))
        Xji1 = Xji*(math.exp(((n - N) / N) * g - ((n + N) / N) * h)
                    - math.exp(((n - N - 1) / N) * g - ((n + N - 1)/N) * h))
        Xij2 = Xij*(math.exp(((n - 1) / N) * g - ((2*N - n + 1) / N) * h)
                    - math.exp((n / N) * g - ((2*N - n)/N) * h))
        Xji2 = Xji*(math.exp(((n - N) / N) * (g + h))
                    - math.exp(((n - N - 1) / N) * (g + h)))
        Yn = eta*(alpha + u)*(Xij1 + Xji1) + eta*(alpha - u)*(Xij2 + Xji2) 
        return Yn
