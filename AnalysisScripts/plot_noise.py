from __future__ import division
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import time
sys.path.append('../PyMKID-master')
import PyMKID_USRP_functions as puf
import PyMKID_resolution_functions as prf
import h5py
from scipy.signal import decimate

series = "20220317_213556"
path   = os.path.join("/data/USRP_Noise_Scans",series.split("_")[0],series)
os.chdir(path)

objects = sorted(os.listdir(os.getcwd()))
noise_files = []
vna_files = []
pulsing_files = []
for fm in range(len(objects)):
    if objects[fm][-3:] == '.h5':
        if 'USRP_Noise_' in objects[fm]:
            noise_files += [objects[fm][:-3]]
        elif '_VNA_' in objects[fm]:
            vna_files += [objects[fm][:-3]]
        elif 'USRP_Noise' in objects[fm]:
            pulsing_files += [objects[fm][:-3]]
print( noise_files)

proper_decimation = 10000
down_sampling = 10000
time_start = 20
time_end = 50
resonance = 4245
near = 1

#with h5py.File('noise_averages.h5', 'r') as fyle:
#    char_fs = np.array(fyle['frequencies'])
#    char_zs = np.array(fyle['means'])

if len(vna_files) == 1:
    vna_files = vna_files*len(noise_files)

i = 0
for noise_file, vna_file in zip(noise_files,vna_files):
    # i += 1
    # if i != 3:
    #     continue


    VNA_f, VNA_z,_ = puf.read_vna(vna_file + '.h5')

    with h5py.File(noise_file + '.h5', "r") as fyle:
        raw_noise = puf.get_raw(fyle)
        amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
        # print(fyle["raw_data0/A_RX2"].attrs.get('freq'))
        decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

    # print(VNA_f,search_freqs)


    near_this_res = np.logical_and(VNA_f > resonance - near, VNA_f < resonance + near)
    VNA_f = VNA_f[near_this_res]
    VNA_z = VNA_z[near_this_res]

    eff_rate = rate/decimation

    total_idx = len(raw_noise)
    total_time = total_idx/eff_rate
    idx_start = int(time_start*eff_rate)
    idx_end = int(time_end*eff_rate)

    noise_window = raw_noise[idx_start:idx_end]
    radius_data, arc_length_data = prf.electronics_basis(raw_noise,axis_option='multiple freqs')

    time = np.linspace(1/eff_rate,total_time,total_idx)
    time_window = time[idx_start:idx_end]

    noise_decimated = prf.average_decimate(raw_noise,proper_decimation)
    radius_decimated = prf.average_decimate(radius_data,proper_decimation)
    arc_decimated = prf.average_decimate(arc_length_data,proper_decimation)
    print(noise_decimated.shape,radius_decimated.shape,arc_decimated.shape)

    eff_rate_dec = eff_rate/proper_decimation
    total_idx_dec = len(noise_decimated)
    total_time = total_idx_dec/eff_rate_dec
    idx_start_dec = int(time_start*eff_rate_dec)
    idx_end_dec = int(time_end*eff_rate_dec )

    time_decimated = time[::proper_decimation]
    time_window_decimated = time_window[::proper_decimation]

    noise_window_decimated = noise_decimated[idx_start_dec:idx_end_dec]
    radius_window_decimated = radius_decimated[idx_start_dec:idx_end_dec]
    arc_window_decimated = arc_decimated[idx_start_dec:idx_end_dec]

    # plt.figure(5)
    # plt.title('radius timestream downsampled')
    # plt.plot(time[idx_start:idx_end:down_sampling],\
    #          radius_data[::down_sampling])
    real_part = raw_noise.real
    imag_part = raw_noise.imag

    plt.figure(1)
    plt.title('S21')
    plt.plot(noise_decimated.real,\
             noise_decimated.imag,alpha = 0.05,marker='.',ls='')
    plt.plot(np.mean(real_part,axis=0),np.mean(imag_part,axis=0),\
             marker='.',ls='',markersize=10,color='k')
    plt.plot(VNA_z.real,VNA_z.imag,'k',ls='',marker='.',alpha=0.05)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axvline(x=0, color='gray')
    plt.axhline(y=0, color='gray')
    plt.legend



    # plt.title('arc length timestream downsampled')
    # plt.plot(time[idx_start:idx_end:down_sampling],\
    #          arc_length_data[::down_sampling])
    arc_std = np.std(arc_window_decimated[:,0])
    num_freqs = int(len(search_freqs))
    visual_separation = np.linspace(0,10*num_freqs*arc_std,num_freqs)

    # plt.figure(3)
    # plt.title('arc length timestream decimated')
    # plt.plot(time_window_decimated,arc_window_decimated + visual_separation)
    #
    plt.figure(4)
    plt.title('radius timestream decimated')
    plt.plot(time_window_decimated,radius_window_decimated + visual_separation)







    # real_mean = np.mean(real_part,axis=0)
    # imag_mean = np.mean(imag_part,axis=0)
    # mean = real_mean[0] + 1j*imag_mean[0]
    #
    # VNA_fr_real = VNA_z[f_idx].real
    # VNA_fr_imag = VNA_z[f_idx].imag
    # VNA_fr_z = VNA_fr_real + 1j*VNA_fr_imag
    #
    # rotation = np.exp(1j*np.angle(mean/VNA_fr_z))
    #
    # VNA_z = VNA_z*rotation


    #f_idx = puf.find_closest(VNA_f,search_freqs[0]*1e-6)
    #puf.plot_noise_and_vna(noise_window_decimated,VNA_z,f_idx,char_zs[i,:])
    plt.show()
