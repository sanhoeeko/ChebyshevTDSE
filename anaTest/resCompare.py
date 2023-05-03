import numpy as np
import analytic as ana
import matplotlib.pyplot as plt
from math import *
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

rhos = np.load('rhos.npy')
mode = 1

'''
plt.plot(rhos.T)
plt.show()
'''

us = []
vs=[]
rhos = rhos.reshape((256, 256, 200))
h = 256

if mode == 0:
    x = 240 / h
    y = 240 / h
    r_a = []
    rhs = []
    for t in range(200):
        tt = 2 * (t + 1) / 3 / pi / 200
        rho_ana = abs(ana.g(x, tt) * ana.g(y, tt) / 5 / h) ** 2
        rho = rhos[240, 240, t]
        r_a.append(rho_ana)
        rhs.append(rho)
    plt.plot(range(200), r_a, range(200), rhs)
    plt.show()
else:
    for t in range(200):
        rho_ana = ana.gen_data(2 * (t + 1) / 3 / pi / 200)

        rho = rhos[:, :, t]
        dif = abs(rho - rho_ana)
        argmax_dif = np.argmax(dif)
        u = dif[argmax_dif // 256, argmax_dif % 256] / rho_ana[argmax_dif // 256, argmax_dif % 256]

        part=rho[64:192,64:192]
        part_ana=rho_ana[64:192,64:192]
        part_dif = abs(part - part_ana)
        argmax_p=np.argmax(part)
        v=part_dif[argmax_p // 128, argmax_p % 128] / part_ana[argmax_p // 128, argmax_p % 128]

        us.append(u)
        vs.append(v)
        print(t)
    plt.plot(range(200), us, range(200), vs)
    plt.legend(['全局相对误差','中心相对误差'])
    plt.show()
