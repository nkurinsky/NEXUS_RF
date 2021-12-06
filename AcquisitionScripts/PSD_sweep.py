from __future__ import division
import sys,os
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
import h5py
#import PyMKID_USRP_functions as PUf
sys.path.append('PyMKID')
import PyMKID_resolution_functions as Prf
#import scipy.signal as sig
import time


objects = sorted(os.listdir(os.getcwd()))
noise_files = []
vna_files = []
calibration_file = None
for fm in range(len(objects)):
    if objects[fm][-3:] == '.h5':
        if 'USRP_Noise_2020' in objects[fm]:
            noise_files += [objects[fm]]
        elif '_VNA_' in objects[fm]:
            vna_files += [objects[fm]]
        elif 'noise_averages' in objects[fm]:
            calibration_file = objects[fm]

if calibration_file != None:
    with h5py.File(calibration_file, 'r') as fyle:
        char_zs = np.array(fyle['means'])
        char_fs = np.array(fyle['frequencies'])
else:
    char_data = None
    char_f = None
char_per_freq = 1
p_ints = []
dis_vars = []
freq_vars = []
# fig_0, axes_0 = plt.subplots(2,1,sharex=True,sharey='row')
# axes_0 = np.expand_dims(axes_0,axis=1)
# print(axes_0[0,0])

for i in range(len(noise_files)):
    if i == i:
        t_init = time.time()
        noise_data = noise_files[i]
        VNA_data = vna_files[int(i/char_per_freq)]
        if calibration_file != None:
            char_data = char_zs[int(i/char_per_freq)]
            char_f = char_fs[int(i/char_per_freq)]
        print(noise_data,VNA_data)
        p_int,f,P_dissipation_clean,P_frequency_clean= Prf.PSDs_and_cleaning(noise_data,VNA_data,char_data,char_f,extra_dec=500)
    dissipation_variance = sum(P_dissipation_clean[1:]) * f[1]
    frequency_variance = sum(P_frequency_clean[1:]) * f[1]
    p_ints.append(p_int)
    dis_vars.append(dissipation_variance)
    freq_vars.append(frequency_variance)
    # if i >= 0 and i <=1:
    #     Prf.plot_PSDs(f,P_dissipation_clean,P_frequency_clean,noise_data,\
    #                   directions=['dissipation','frequency'],\
    #                   units=['d(1/Q)','df/f'],savefig='resonator_across_powers',\
    #                   fig_0=fig_0,axes_0=axes_0)
    #     print '  '+str(round(time.time() - t_init,2))+' s'

# plt.show(False)


# print(p_ints,dis_vars,freq_vars)
plt.figure('noise vs internal power')
plt.semilogy(p_ints,dis_vars,ls='',marker='s',label='d(1/Q) niobium',color='C0')
plt.semilogy(p_ints,freq_vars,ls='',marker='s',label='df/f niobium',color='C1')
plt.legend()
plt.ylabel('sum of discrete PSD (excluding bin 0) * df')
plt.xlabel('internal power (dB wrt to unity at the USRP)')
plt.title('Noise in resonator basis vs internal power')
plt.grid(True)
plt.show()
print('hello')
