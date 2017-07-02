#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

popsi = open("P_Opsi.dat", "rb")

# TODO: automatically find these values in two_dim_global.f90
Ny = 280
Nz = 450
snap = 1001

def phase_frame(i):
    popsi.seek(Ny * Nz * 16 * i)
    frame = np.fromfile(popsi, dtype=np.float64, count=Ny * Nz * 2)
    re = frame[0::2]
    im = frame[1::2]
    phase = np.angle(re + im * 1j)
    grid = phase.reshape((Ny, Nz))
    return grid

fig = plt.figure()

i = 0
p = plt.imshow(phase_frame(0), cmap=cm.gist_rainbow)

def updatefig(*args):
    global i
    i += 1
    if i == snap:
        i = snap - 1
    p.set_data(phase_frame(i))
    return p,

ani = animation.FuncAnimation(fig, updatefig, interval=20, blit=True)

plt.show()

popsi.close()

