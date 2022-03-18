from __future__ import division
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import time
# sys.path.append('../PyMKID-master')
import PyMKID_USRP_functions as puf
import PyMKID_resolution_functions as prf
import h5py
from scipy.signal import decimate

# noise_directory = '201227 noise/4p240/'
# noise_directory = '220201 laser data/-30/0 NDF 3p0V AWF/'
# noise_directory = '220202 laser data/-30/0 NDF 2p8V AWF/'
series = "20220317_223034"
noise_directory = os.path.join('/data/USRP_Noise_Scans',series.split('_')[0],series)

objects = sorted(os.listdir(noise_directory))
# print(objects)
noise_noise_files = []
noise_vna_files = []
noise_calibration_file = None
for fm in range(len(objects)):
    if objects[fm][-3:] == '.h5':
        if '_Noise_' in objects[fm]:
            # noise_noise_files += [noise_directory + objects[fm]]
            noise_noise_files = np.append(noise_noise_files, os.path.join(noise_directory,objects[fm]))
        elif '_VNA_' in objects[fm]:
            # noise_vna_files += [noise_directory + objects[fm]]
            noise_vna_files = np.append(noise_vna_files, os.path.join(noise_directory,objects[fm]))
        elif 'noise_averages' in objects[fm]:
            # noise_calibration_file = noise_directory + objects[fm]
            noise_calibration_file = np.append(noise_calibration_file, os.path.join(noise_directory,objects[fm]))
# noise_noise_files = noise_noise_files[1:2]
print(noise_noise_files)

proper_decimation = 1
down_sampling = 1000
time_start = 3
time_end = 9.9
resonance = 4240.1
near = 1000

# with h5py.File('noise_averages.h5', 'r') as fyle:
#     char_fs = np.array(fyle['frequencies'])
#     char_zs = np.array(fyle['means'])

# if len(vna_files) == 1:
#     vna_files = vna_files*len(noise_files)

i = 0
for noise_file, vna_file in zip(noise_noise_files,noise_vna_files):
    if i == i: #len(noise_noise_files) - 1:
        VNA_f, VNA_z, _ = puf.read_vna(vna_file)

        try:
            with h5py.File(noise_file, "r") as fyle:
                raw_noise = puf.get_raw(fyle)
                amplitude = fyle["raw_data0/B_TXRX"].attrs.get('ampl')
                rate = fyle["raw_data0/B_RX2"].attrs.get('rate')
                LO = fyle["raw_data0/B_RX2"].attrs.get('rf')
                search_freqs = fyle["raw_data0/B_RX2"].attrs.get('rf') + fyle["raw_data0/B_RX2"].attrs.get('freq')
                # print(fyle["raw_data0/A_RX2"].attrs.get('freq'))
                decimation = fyle["raw_data0/B_RX2"].attrs.get('decim')
        except:
            with h5py.File(noise_file, "r") as fyle:
                raw_noise = puf.get_raw(fyle)
                amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
                rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
                LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
                search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
                # print(fyle["raw_data0/A_RX2"].attrs.get('freq'))
                decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')


        # print(VNA_f,search_freqs)
        # with h5py.File(noise_calibration_file, 'r') as fyle:
        #     noise_char_zs = np.array(fyle['means'])
        #     noise_char_fs = np.array(fyle['frequencies'])




        near_this_res = np.logical_and(VNA_f > resonance - near, VNA_f < resonance + near)
        # VNA_f = VNA_f[near_this_res]
        # VNA_z = VNA_z[near_this_res]

        eff_rate = rate/decimation

        total_idx = len(raw_noise)
        total_time = total_idx/eff_rate
        idx_start = int(time_start*eff_rate)
        idx_end = int(time_end*eff_rate)

        noise_window = raw_noise[idx_start:idx_end,:]
        radius_data, arc_length_data,radius_means,_ = prf.electronics_basis(raw_noise,axis_option='multiple freqs')

        time = np.linspace(1/eff_rate,total_time,total_idx)
        time_window = time[idx_start:idx_end]


        noise_decimated = prf.average_decimate(raw_noise,proper_decimation)
        radius_decimated = prf.average_decimate(radius_data,proper_decimation)
        arc_decimated = prf.average_decimate(arc_length_data,proper_decimation)
        # print(noise_decimated.shape,radius_decimated.shape,arc_decimated.shape)

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
        # print(arc_window_decimated[100:105])

        # plt.figure(5)
        # plt.title('radius timestream downsampled')
        # plt.plot(time[idx_start:idx_end:down_sampling],\
        #          radius_data[::down_sampling])
        real_part = raw_noise.real
        imag_part = raw_noise.imag

        # plt.figure(1)
        # plt.title('S21 downsampled')
        # plt.plot(real_part[idx_start:idx_end:down_sampling],\
        #          imag_part[idx_start:idx_end:down_sampling],alpha = 0.05,marker='.',ls='')
        # plt.plot(np.mean(real_part,axis=0),np.mean(imag_part,axis=0),\
        #          marker='.',ls='',markersize=10,color='k')
        # plt.plot(VNA_z.real,VNA_z.imag,'k')
        #
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.axvline(x=0, color='gray')
        # plt.axhline(y=0, color='gray')

        # plt.figure(2)
        # plt.title('arc length timestream downsampled')
        # plt.plot(time[idx_start:idx_end:down_sampling],\
        #          arc_length_data[::down_sampling])
        arc_std = np.std(arc_window_decimated[:,0])
        radius_std = np.std(radius_window_decimated[:,0])
        num_freqs = int(len(search_freqs))
        visual_separation_arc = np.linspace(0,10*num_freqs*arc_std,num_freqs)
        visual_separation_radius = np.linspace(0,10*num_freqs*radius_std,num_freqs)


        # puf.plot_noise_and_vna(noise_window_decimated,VNA_z,char_zs=noise_char_zs[[i],:],alpha=0.1)
        puf.plot_noise_and_vna(noise_window,VNA_z,alpha=0.1)


        plt.figure(3)
        plt.title('arc length timestream decimated')
        plt.plot(time_window_decimated,arc_window_decimated + visual_separation_arc)
        plt.xlabel('time (s)')
        plt.ylabel('ADC units')

        plt.figure(4)
        plt.title('radius timestream decimated')
        plt.plot(time_window_decimated,radius_window_decimated + visual_separation_radius)
        plt.xlabel('time (s)')
        plt.ylabel('ADC units')

        VNA_dB = 20*np.log10(abs(VNA_z))
        noise_dB = 20*np.log10(np.mean(abs(noise_window),axis=0))
        # noise_char_dB = 20*np.log10(abs(noise_char_zs))
        plt.figure('absolute value')
        plt.plot(VNA_f*1e6,VNA_dB)
        plt.plot(search_freqs,noise_dB,ls='',marker='.',markersize=10,label='noise timestream average')
        #plt.plot(noise_char_fs,noise_dB,ls='',marker='.',markersize=10,label='noise timestream average')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('dB relative to unity at the USRP')
        plt.legend()


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


        # f_idx = puf.find_closest(VNA_f,search_freqs[0]*1e-6)
        # char_zs = np.squeeze(char_zs)

    i += 1
    # plt.show(False)
    # raw_input('press enter to close')
    # plt.close('all')

    plt.show()
