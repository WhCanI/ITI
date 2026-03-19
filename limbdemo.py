# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:51:24 2025

@author: jkf
"""

import sim2 as sim
from matplotlib import pyplot as plt

im,h=sim.fitsread('/home/jhpan/fits_file/filament/19200106/HA_19200106T092200_cal.fits')

plt.figure()
sim.showim(im)

im1=sim.removelimb(im)[0]
plt.figure()
sim.showim(im1)

plt.show()
