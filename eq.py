import numpy as np
import matplotlib.pyplot as plt
import math

k=1
N=1000
step=100
x_ini=0.
x_end=np.pi
a = 1.4
dx=(x_end - x_ini)/N
dx2=dx**2


def IC():
    global u
    u=np.zeros(N+1)
    u[0]=0.0
    u[N]=0.0
    # IC
    for i in range(1,N):
        x = x_ini + i*dx
        if x_ini<x and x<=a:
            u[i]=0. * x
        elif a<x and x<x_end:
            u[i]= (x-a)*(x_end-x)
    M=np.linspace(x_ini,x_end,N+1)
    #plt.plot(M,u)
    #plt.pause(1.)


def u_m(m,x,t):
    for i in range(len(x)):
        try:
            x[i] = x[i] + a
        except:
            x[i] = 0.
    return (2/np.pi)*(1/m**3)*((-1)**m - np.cos(m*a) + m*(a-np.pi)*np.sin(m*a))*np.sin(m*x)*np.exp(-(m**2)*t)

if __name__=='__main__':
    IC()
    plt.plot(u)
    plt.show()
    M=np.linspace(x_ini,x_end,N+1)
    for t in range(0,10,1):
        u=np.zeros(N+1)
        print(t)
        for m in range(1,5): # m=3
            print(m)
            u = u_m(m,M,t/5)
            plt.plot(M,u)
            plt.pause(0.1)
    plt.show()
