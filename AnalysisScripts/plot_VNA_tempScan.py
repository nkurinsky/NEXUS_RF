from __future__ import division
import sys,os
sys.path.append('/home/nexus-admin/workarea/PyMKID')
import PyMKID_USRP_functions as puf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import PyMKID_USRP_functions as puf
from glob import glob

datapath='/data/TempSweeps/'
day='20201222'
series='20201222_152824'
power='-50'

vna_files = glob(datapath+day+'/*P'+power+'*'+series+'.h5')
vna_files.sort(key=os.path.getmtime)

#plt.figure(1)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.axvline(x=0, color='gray')
#plt.axhline(y=0, color='gray')

fname = vna_files[-1]
print(fname)

plt.rcParams.update({'font.size': 15})
norm = plt.Normalize(vmin=80,vmax=250)
for fname in vna_files:

    raw_f, raw_VNA, amplitude = puf.read_vna(fname)
    #power = -14 + 20*np.log10(amplitude)

    temp = fname.split('_')[1][1:]
    color = cm.jet(norm(float(temp)))
    
    plt.figure(2,figsize=(8,6))
    plt.plot(raw_f,20*np.log10(abs(raw_VNA)),label=temp+' mK',color=color)

    #plt.figure(1)
    #plt.plot(raw_VNA[::10].real,raw_VNA[::10].imag)

plt.title(('MKID Frequency Sweep at ' +power+' dBm'), fontdict = {'fontsize': 18})
plt.figure(2)
plt.xlabel('f [MHz]', fontdict = {'fontsize': 18})
plt.ylabel('S21 [dB]', fontdict = {'fontsize': 18})
cbar=plt.colorbar(cm.ScalarMappable(cmap=cm.jet, norm=norm),shrink=0.8)
cbar.set_label('Temperature [mK]', size=16)
#plt.legend(loc='center left',bbox_to_anchor=(1.,0.5))
plt.tight_layout()
plt.show()
