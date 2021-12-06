from __future__ import division
import sys,os
#sys.path.append('/home/nexus-admin/workarea/PyMKID')
#import PyMKID_USRP_functions as puf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
#import PyMKID_USRP_functions as puf
from glob import glob

#import file reading function
sys.path.append('/home/nexus-admin/NEXUS_RF/Devices')
from VNAfunctions import *
vna=VNA()

#Define plotted data power and datetime
datapath='/data/TempSweeps/VNA/'
day='20210118'
series='20210118_201354'
power='-45'
#Differentiates between version that handles complex data or not (new or old)
new_version = False

vna_files = glob(datapath+day+'/*P'+power+'*'+series+'.txt')
vna_files.sort(key=os.path.getmtime)

#plt.figure(1)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.axvline(x=0, color='gray')
#plt.axhline(y=0, color='gray')

fname = vna_files[-1]
print(fname)

norm = plt.Normalize(vmin=85,vmax=250)
if new_version:
    for fname in vna_files:

        #raw_f, raw_VNA, power = puf.read_vna(fname)
        freqs, real, imag = vna.readData(fname) 

        temp = fname.split('_')[1][1:]
        color = cm.jet(norm(float(temp)))

        mags, angles = vna.comp2mag(real,imag)

        plt.figure(2,figsize=(8,6))
        plt.plot(freqs,mags,label=temp+' mK',color=color)
        plt.title(str(power)+' dBm')

    #plt.figure(1)
    #plt.plot(raw_VNA[::10].real,raw_VNA[::10].imag)

else:
    for fname in vna_files:

        #raw_f, raw_VNA, power = puf.read_vna(fname)
        freqs, mags = vna.readData_old(fname)

        temp = fname.split('_')[1][1:]
        color = cm.jet(norm(float(temp)))

        #mags, angles = vna.comp2mag(real,imag)

        plt.figure(2,figsize=(8,6))
        plt.plot(freqs,mags,label=temp+' mK',color=color)
        plt.title(str(power)+' dBm')

plt.figure(2)
plt.xlabel('f [MHz]')
plt.ylabel('S21 [dB]')
plt.colorbar(cm.ScalarMappable(cmap=cm.jet, norm=norm),shrink=0.8,label='Temperature (mK)')
#plt.legend(loc='center left',bbox_to_anchor=(1.,0.5))
plt.tight_layout()
plt.show()
