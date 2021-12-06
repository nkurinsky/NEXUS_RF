from __future__ import division
import sys,os
sys.path.append('PyMKID')
import numpy as np
import matplotlib.pyplot as plt
import time
import PyMKID_USRP_functions as puf

objects = sorted(os.listdir(os.getcwd()))
noise_files = []
vna_files = []
for fm in range(len(objects)):
    if objects[fm][-3:] == '.h5':
        if 'Noise' in objects[fm]:
            noise_files += [objects[fm][:-3]]
        elif 'USRP_VNA_' in objects[fm]:
            vna_files += [objects[fm][:-3]]

# powers = [-40]# dBm out of USRP

plt.figure(1)
plt.gca().set_aspect('equal', adjustable='box')
plt.axvline(x=0, color='gray')
plt.axhline(y=0, color='gray')

print(vna_files)
for Vn in range(len(vna_files)):
    raw_f, raw_VNA, amplitude = puf.read_vna(vna_files[Vn]+'.h5')

    power = -14 + 20*np.log10(amplitude)
    #target_f0 = 3518
    #target_f1 = 3525
    #target_f0_arg = np.argmin(abs(raw_f-target_f0))
    #target_f1_arg = np.argmin(abs(raw_f-target_f1))
    #res_nearness = np.min(abs(np.tile(reses,(len(raw_f),1)).T-raw_f),axis=0)
    plt.figure(2)
    plt.plot(raw_f,20*np.log10(abs(raw_VNA)),label=str(power)+' dBm')
    #plt.plot(raw_f[res_nearness>2][::10],20*np.log10(abs(raw_VNA[res_nearness>2][::10])))
    #plt.plot(raw_f[target_f0_arg:target_f1_arg],20*np.log10(abs(raw_VNA[target_f0_arg:target_f1_arg])),'k')
    plt.figure(1)
    plt.plot(raw_VNA[::10].real,raw_VNA[::10].imag)
    #plt.plot(raw_VNA[res_nearness>2][::10].real,raw_VNA[res_nearness>2][::10].imag)
    #plt.plot(raw_VNA[target_f0_arg:target_f1_arg].real,raw_VNA[target_f0_arg:target_f1_arg].imag,'k')

plt.figure(2)
plt.xlabel('f [MHz]')
plt.ylabel('S21 [dB]')
plt.legend(loc='center left',bbox_to_anchor=(1.,0.5))
plt.show()
