from __future__ import division
import sys,os
sys.path.append('PyMKID')
import numpy as np
import matplotlib.pyplot as plt
import time
import PyMKID_USRP_functions as puf
from glob import glob

vna_files = glob('USRP_VNA*.h5')
vna_files.sort(key=os.path.getmtime)

plt.figure(1)
plt.gca().set_aspect('equal', adjustable='box')
plt.axvline(x=0, color='gray')
plt.axhline(y=0, color='gray')

fname = vna_files[-1]
print(fname)

raw_f, raw_VNA, amplitude = puf.read_vna(fname)
power = -14 + 20*np.log10(amplitude)

plt.figure(2)
plt.plot(raw_f,20*np.log10(abs(raw_VNA)),label=str(power)+' dBm')

plt.figure(1)
plt.plot(raw_VNA[::10].real,raw_VNA[::10].imag)

plt.figure(2)
plt.xlabel('f [MHz]')
plt.ylabel('S21 [dB]')
plt.legend(loc='center left',bbox_to_anchor=(1.,0.5))
plt.show()
