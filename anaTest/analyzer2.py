import ctypes as c
import numpy as np

# import matplotlib.pyplot as plt

data = np.zeros((256 * 256, 200))


# plt.ion()


def trans_data(n):
    for i in range(256 * 256):
        data[i, n] = ptr[i]


dll = c.CDLL("Cheby_dll_test.dll")
dll.set_timeSpan.argtypes = [c.c_double]
dll.getPeriod.restype = c.c_double
dll.analytic_test_single.restype = c.c_double
dll.analytic_test_init.restype = c.POINTER(c.c_double)
ptr = dll.analytic_test_init()
period = dll.getPeriod()
dt = period / 200
dll.set_timeSpan(dt)

###############################
rs = []
for t in range(200):
    r = dll.analytic_test_single()
    trans_data(t)
    print('t=', t, '\t r=', r)
    rs.append(r)

###############################

np.save('rhos.npy', data)
np.save('rs.npy', np.array(rs))
dll.end_test()
