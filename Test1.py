import math
import numpy as np
import matplotlib.pyplot as plt

def GSinverse(t, Qs, omega):
    #finding the kappa coefficients
    if(t==0):
        raise(ValueError("t=0 is not a valid input"));
    kappa=[0]*omega;
    for n in range (1,omega+1):
        omega2=omega//2;
        sign= 1 if (n+omega2==0) else -1;
        lowK=math.floor((n+1)/2);
        highK=min(n,omega2);
        Sum=0;
        for k in range(lowK,highK):
            summand=k**omega2/math.factorial(omega2-k);
            summand=summand*(math.comb(2*k, k)*math.comb(k,n-k));
            Sum=Sum+summand;
        kappa[n]=sign*Sum;
    #Computing the inverse Laplace at t
    log2t=math.log(2)/t;
    qt=0;
    for n in range(1,omega+1):
        qt=qt+kappa[n]*(Qs(n*log2t));
    qt=log2t*qt;
    return qt;


fs = lambda s: 1/(s+1)**2
ft = lambda t: t*math.exp(-t)

delta=0.1;
t=np.arange(delta,10,delta)
f=[None]*len(t);
for i in range(len(t)):
    time=delta*(i+1);
    f[i]=GSinverse(time,fs,10);

plt.plot(t,ft(t));
plt.plot(t,f);

