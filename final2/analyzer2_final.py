import matplotlib
matplotlib.use('Agg') #spyder

import ctypes as c
import numpy as np
import math as mh
import matplotlib.pyplot as plt

data = np.zeros((1024 * 512,))


def trans_data():
    for i in range(1024 * 512):
        data[i] = ptr[i]
    return data.reshape((1024, 512))


dll = c.CDLL("Cheby_dll_final.dll")
dll.set_timeSpan.argtypes = [c.c_double]
dll.set_momentum.argtypes = [c.c_double]
dll.set_sigma.argtypes = [c.c_double]
dll.main_proj_single.restype = c.c_double
dll.main_proj_init.restype = c.POINTER(c.c_double)

dll.set_iter(200)
dll.set_timeSpan(50.0)
dll.set_momentum(256 * mh.pi)
dll.set_sigma(0.1)
dll.seeParams()
ptr = dll.main_proj_init()

###############################

rs = []
for t in range(100):
    r = dll.main_proj_single()
    print('rho_tot=', r)
    res = trans_data()
    plt.imshow(res)
    plt.savefig('pics/' + str(t) + '.png')
    rs.append(r)
    print('t=', t)

###############################
rs = np.array(rs)
np.save('rs.npy', rs)
dll.end_test()
