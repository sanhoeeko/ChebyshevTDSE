import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath as cm

w1=pi**2
h=256

def g(x,t):
    return sin(pi*x)*cm.exp(-1j*w1*t)-3*sin(2*pi*x)*cm.exp(-1j*4*w1*t)

def gen_data(t):
    mat=np.zeros((h,h))
    for i in range(h):
        for j in range(h):
            x=i/h; y=j/h
            mat[i,j]=abs(g(x,t)*g(y,t)/5/h)**2
    return mat


if __name__=='__main__':
    t=0.8
    mat=gen_data(2*t/3/pi)
    plt.imshow(mat)
    plt.show()