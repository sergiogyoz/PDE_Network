# %%
import math
import numpy as np
import matplotlib.pyplot as plt
# My module
from pde_network import InitialConditions
from pde_network import _f_tools
from pde_network import Solver
print("loaded libraries")
# %%
# ----- Testing the example 1, Fig 4 from the paper
# value parameters units are in mm, micromoles and seconds
k = 5 *(10**-3)  # mmole(milimole)/litre --> micromole/mm^3
F = 0.002  # mm^3/s (regarless of S_i)
rho = 0.05  # mm/s density of transporters
Dm = 6.7*(10**-4)  # mm^2/s molecular diff coeff of glucose in water at body temp
l12 = 1  # mm
Srange = [sval*(10**-6) for sval in range(100,1001,100)]  #  micron^2 --> mm^2 for 1<=k<=10
n = 2  # size of the network
Ngrid = 101
Omega = 6
# storage variable
C = []
for S12 in Srange:
    # Networks parameters
    qx = np.zeros([n, n, Ngrid])

    S = np.zeros([n,n], dtype=np.float64)
    S[0,1] = S12
    InitialConditions.symmetrize(S)

    l = np.zeros([n,n], dtype=np.float64)
    l[0,1] = l12
    InitialConditions.symmetrize(l)

    R = np.zeros([n,n], dtype=np.float64)
    R[0,1] = rho/math.sqrt(S12)
    InitialConditions.symmetrize(R)

    u = np.zeros([n,n], dtype=np.float64)
    u[0,1] = F/S12  # F=u*S
    InitialConditions.antisymmetrize(u)

    D = np.zeros([n,n], dtype=np.float64)
    def Dij(i, j):
        rij = math.sqrt(S[i,j]/math.pi)
        return Dm + u[i,j]*u[i,j]*rij*rij/(48*Dm)
    D[0,1] = Dij(0,1)
    InitialConditions.symmetrize(D)

    Fp = (F/2) * (1 + math.sqrt(1
                              + 4 * R[0,1] * Dm * S[0,1]*S[0,1] / (F*F)
                              + S[0,1] / (48 * math.pi * Dm)))

    # initially empty network so it's characterized with the propagation matrix
    I_dummy = [_f_tools.zerof for i in range(2)]
    ic = InitialConditions(qx,l,R,D,u,S,I_dummy)
    solver = Solver(ic)
    M = solver.prog_matrix(0)
    Ctot = k * (M[0,0]
                + (M[1,0]*Fp-M[1,0]*M[0,1])
                / (M[1,1]+Fp))

    C.append(Ctot)
    print(f"Finished S = {S12:.0e}")

for parentesis in [1]:
    # plot the sampled points
    xscale = 10**6  # mm^2 to microns^2
    x = [xscale*S12 for S12 in Srange]
    yscale = 60*60  # micromole/seconds --> micromole/hours
    y = [yscale*Ctot for Ctot in C]
    plt.figure(figsize=(6,6), dpi=150)
    plt.plot(x, y,
            label = "fixed current, numerical solution",
            linestyle = "None",
            marker = "+",
            markersize = 12)
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.xlabel("Cross sectional area of the vessel, $\mu m^2$")
    plt.ylabel("Total rate of resource consumption, $\mu$mole per hour")

# %%
for parentesis in [1]:
    M = np.zeros((2,2),dtype=np.float64)
    l = 1
    F = 0.002
    rho = 0.05
    Dm = 6.7*(10**-4)
    k = 5 *(10**-3)
    Srange = [sval*(10**-6) for sval in range(1,1000)]
    C = []
    for S in Srange:
        r = math.sqrt(S/math.pi)
        u = F/S
        D = Dm + u*u*r*r/(48*Dm)
        R = rho / math.sqrt(S)
        alpha = math.sqrt(u*u + 4*D*R)
        Fp = (F/2) * (1 + math.sqrt(1
                            + 4 * R * Dm * S*S / (F*F)
                            + S / (48 * math.pi * Dm)))
        h = alpha*l / (2*D)
        g = u*l / (2*D)
        M[0,0] = (S/2) * (u + alpha/math.tanh(h))
        M[1,1] = (S/2) * (-u + alpha/math.tanh(h))
        M[0,1] = (-S/2) * (alpha*math.exp(-g) / math.sinh(h))
        M[1,0] = (-S/2) * (alpha*math.exp(g) / math.sinh(h))
        Ctot = k * (M[0,0]
                    + (M[1,0]*Fp-M[1,0]*M[0,1])
                    / (M[1,1]+Fp))

        C.append(Ctot)

    xscale = 10**6  # mm^2 to microns^2
    x = [xscale*S12 for S12 in Srange]
    yscale = 60*60  # micromole/seconds --> micromole/hours
    y = [yscale*Ctot for Ctot in C]
    plt.plot(x, y,
             linestyle="--",
             label = "fixed current, analytic solution")
    plt.legend()

# %%
for parentesis in [1]:
    M = np.zeros((2,2),dtype=np.float64)
    l = 1
    mu = 0.007*(10**-1)  # g/(cm*s) --> g/(mm*s) viscosity of water
    rho = 0.05
    Dm = 6.7*(10**-4)
    k = 5 *(10**-3)
    Srange = [sval*(10**-6) for sval in range(1,1000)]
    DeltaP = 8*math.pi*mu*l*(2)/Srange[-1]
    C = []
    for S in Srange:
        r = math.sqrt(S/math.pi)
        u = S* (DeltaP/(8*math.pi*mu*l))
        D = Dm + u*u*r*r/(48*Dm)
        R = rho / math.sqrt(S)
        alpha = math.sqrt(u*u + 4*D*R)
        F = u*S
        Fp = (F/2) * (1 + math.sqrt(1
                    + 4 * R * Dm * S*S / (F*F)
                    + S / (48 * math.pi * Dm)))
        h = alpha*l / (2*D)
        g = u*l / (2*D)
        M[0,0] = (S/2) * (u + alpha/math.tanh(h))
        M[1,1] = (S/2) * (-u + alpha/math.tanh(h))
        M[0,1] = (-S/2) * (alpha*math.exp(-g) / math.sinh(h))
        M[1,0] = (-S/2) * (alpha*math.exp(g) / math.sinh(h))
        Ctot = k * (M[0,0]
                    + (M[1,0]*Fp-M[1,0]*M[0,1])
                    / (M[1,1]+Fp))

        C.append(Ctot)

    xscale = 10**6  # mm^2 to microns^2
    x = [xscale*S12 for S12 in Srange]
    yscale = 60*60  # micromole/seconds --> micromole/hours
    y = [yscale*Ctot for Ctot in C]
    plt.plot(x, y,
             linestyle="--",
             label = "fixed pressure drop")
    plt.legend()
    plt.show()

# %%
