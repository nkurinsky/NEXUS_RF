from __future__ import division
import sys,os
import time

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm

import pandas as pd
from   glob import glob

#sys.path.append('/home/nexus-admin/workarea/PyMKID')
import PyMKID_USRP_functions as puf
import fitres

## Path to VNA data
dataPath = '/data/PowerSweeps/VNA/'

## Series identifier
day    = '20211204'
time   = '153907'
series = day + '_' + time
srPath = dataPath + day + '/' + series + '/'

## File string format
fn_prefix = "Psweep_P"
fn_suffix = "_" + series + ".txt"

## Find and sort the relevant directories in the series
print("Searching for files in:", srPath)
print(" with prefix:", fn_prefix)
print(" and  suffix:", fn_suffix)
vna_files = glob(srPath+fn_prefix+'*'+fn_suffix)
vna_files.sort(key=os.path.getmtime)
print("Using files:")
for fname in vna_files:
    print("-",fname)

def read_cmt_vna(fname):
    df = pd.read_csv(fname)
    f  = df['freq (Hz)'].to_numpy()
    S21_real = df[' S21 Real'].to_numpy()
    S21_imag = df[' S21 Imag'].to_numpy()
    return f, S21_real, S21_imag

#plt.figure(1)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.axvline(x=0, color='gray')
#plt.axhline(y=0, color='gray')



plt.rcParams.update({'font.size': 15})
norm = plt.Normalize(vmin=80,vmax=250)
fr_list = []; Qr_list = []; Qc_list = []; Qi_list = []; power_list =[]
for fname in vna_files:
    ix_pwr_i = len(srPath+fn_prefix)
    ix_pwr_f = len(fname)-len(fn_suffix)
    power = int(float(fname[ix_pwr_i:ix_pwr_f]))
    print("Extracting data for power:",power,"dBm")
    power_list.append(power)

    f, S21_real, S21_imag = read_cmt_vna(fname)
    z = S21_real + 1j*S21_imag
    f /= 1.0e9

    fr, Qr, Qc, Qi = fitres.sweep_fit(f,z,start_f=f[0],stop_f=f[-1])

    fr_list.append(fr[0]); Qr_list.append(Qr[0])
    Qc_list.append(Qc[0]); Qi_list.append(Qi[0])
    #power = -14 + 20*np.log10(amplitude)

    # temp = fname.split('_')[1][1:]
    # color = cm.jet(norm(float(temp)))
    #
    # plt.figure(2,figsize=(8,6))
    # plt.plot(raw_f,20*np.log10(abs(raw_VNA)),label=temp+' mK',color=color)

    #plt.figure(1)
    #plt.plot(raw_VNA[::10].real,raw_VNA[::10].imag)

plt.figure()
plt.plot(power_list,fr_list)
plt.xlabel('power (dBm)')
plt.ylabel('resonator frequency')

plt.figure()
plt.plot(power_list,Qr_list)
plt.xlabel('power (dBm)')
plt.ylabel('resonator Q')
plt.show()

# plt.title(('MKID Frequency Sweep at ' +power+' dBm'), fontdict = {'fontsize': 18})
# plt.figure(2)
# plt.xlabel('f [MHz]', fontdict = {'fontsize': 18})
# plt.ylabel('S21 [dB]', fontdict = {'fontsize': 18})
# cbar=plt.colorbar(cm.ScalarMappable(cmap=cm.jet, norm=norm),shrink=0.8)
# cbar.set_label('Temperature [mK]', size=16)
# #plt.legend(loc='center left',bbox_to_anchor=(1.,0.5))
# plt.tight_layout()
# plt.show()