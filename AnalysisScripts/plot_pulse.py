# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:19:44 2020

@author: awesm
"""

from __future__ import division
import os
import sys
sys.path.append('../PyMKID-master')
import numpy as np
import matplotlib.pyplot as plt
import PyMKID_USRP_functions as puf
import h5py
import copy

total_time = 30.
series = "20220317_213556"

def idx2time(idx,total_idx=total_time*5000000.,tot_time=total_time):
    return idx/total_idx*tot_time

def time2idx(t,total_idx=total_time*5000000.,tot_time=total_time):
    return int(t/total_time*total_idx)

pulse_period = 0.01
pulse_duration = 20.e-6
plot_duration = 1000.e-6
baseline_duration = 1000.e-6
time = np.linspace(0,plot_duration*1e6,\
                       time2idx(plot_duration))
c_wheel = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']




# plt.figure(1)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.axvline(x=0, color='gray')
# plt.axhline(y=0, color='gray')

def plot_pulse_scans(filename):
##    plt.close('all')
    with h5py.File(filename,'r') as fyle:
        fr_scan_frequencies = np.asarray(fyle['fr scan frequencies'].get('scan frequencies'))
        fr_scans = np.asarray(fyle['fr scan frequencies'].get('S21 of scan frequencies'))
        j = 0
        plt.figure(1)
        plt.plot(fr_scans[:,0].real,fr_scans[:,0].imag,\
                            color = 'b',\
                            ls = '',\
                            marker = '.',\
                            markersize = 20)
        plt.plot(fr_scans[:,1].real,fr_scans[:,1].imag,\
                            color = 'r',\
                            ls = '',\
                            marker = '.',\
                            markersize = 20)
        for pulse_frequency in fyle.keys():
            if pulse_frequency != 'fr scan frequencies':
                pulse_avg = np.asarray(fyle[str(pulse_frequency)].get('average pulse shape'))
                baseline_avg = np.asarray(fyle[str(pulse_frequency)].get('average baseline level'))
                for i in range(np.size(pulse_avg,axis=1)):

                    # trajectory_color = (0.75,\
                    #                     (i+1)/len(fr_scan_frequencies),\
                    #                     1-j/(len(fyle.keys())-1))

                    trajectory_color = c_wheel[j % len(c_wheel)]
                    resonator_idx = int(np.floor(np.size(fr_scan_frequencies,axis=1)/2))

                    plt.figure(1)
                    plt.plot(pulse_avg[:,i].real,pulse_avg[:,i].imag,\
                             color = trajectory_color)
                    plt.plot(baseline_avg[i].real,baseline_avg[i].imag,\
                             color = 'k',\
                             ls = '',\
                             marker = '.',\
                             markersize = 10)
                    plt.plot(fr_scans[i,:].real,fr_scans[i,:].imag,\
                            color = c_wheel[i],\
                            ls = '',\
                            marker = '.',\
                            markersize = 20)
                    plt.plot(fr_scans[i,resonator_idx].real,fr_scans[i,resonator_idx].imag,\
                            color = 'r',\
                            ls = '',\
                            marker = '.',\
                            markersize = 20)

                    # print(pulse_avg)
                    # print(baseline_avg[i])
                    mag_dS21 = abs(pulse_avg[:,i]-baseline_avg[i])


                    plt.figure(2)
                    plt.plot(time,\
                             mag_dS21,\
                             color = trajectory_color,\
                             label = pulse_frequency)

                   # plt.title(pulse_file
                j += 1
    # plt.legend()
    plt.show(False)





def average_traces(noise_file):

    print(noise_file)
    with h5py.File(noise_file, "r") as fyle:
        raw_noise = puf.get_raw(fyle)
        amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
        decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

    # print(raw_noise[:,0])
    # raw_noise_copy = copy.deepcopy(raw_noise)
    # raw_noise_copy[:,0] = raw_noise[:,1]
    # raw_noise_copy[:,1] = raw_noise[:,0]
    # raw_noise = raw_noise_copy
    # print(raw_noise[:,0])
    noise_mag = abs(raw_noise)
    noise_mag_avg = np.mean(noise_mag,axis=0)
    noise_mag_std = np.std(noise_mag,axis=0)

#    print(noise_mag_avg)



    search_timer = 0.1
    search_counter = 0

    avg_counter = 0

    pulse_start_idxs = []
#    pulse_idx = int(np.size(noise_mag,axis=1))-1
    pulse_idx = 1
    pulse_summation = np.zeros((time2idx(plot_duration),len(search_freqs)),dtype=complex)

    baseline_summation = np.zeros(len(search_freqs),dtype=complex)

    while search_timer <= total_time*.95:
        print(search_timer)
        search_idx = time2idx(search_timer)
        if search_counter == 0:
            search_window_size = int(1.01*time2idx(pulse_period))
            noise_mag_window = noise_mag[search_idx:search_idx+search_window_size,pulse_idx]
            window_shift = 0
        else:
            search_half_window_size = int(time2idx(pulse_duration)/2)
            noise_mag_window = noise_mag[search_idx-search_half_window_size:\
                                         search_idx+search_half_window_size,pulse_idx]
            window_shift = search_half_window_size

        elevated_noise_mag = np.argwhere(abs(noise_mag_window - noise_mag_avg[pulse_idx]) > 5*noise_mag_std[pulse_idx])
        pulse_start_idx = search_idx + elevated_noise_mag[0,0] - window_shift


        pulse_start_idxs.append(pulse_start_idx)

        pulse_start_time = idx2time(pulse_start_idx)

        plot_start_idx = pulse_start_idx + time2idx(1.1*pulse_duration)
        plot_stop_idx = plot_start_idx + time2idx(plot_duration)

        baseline_stop_idx = pulse_start_idx - 10
        baseline_start_idx = baseline_stop_idx - time2idx(baseline_duration)
        baseline = np.mean(raw_noise[baseline_start_idx:baseline_stop_idx],axis=0)

        pulse_single_trace = raw_noise[plot_start_idx:plot_stop_idx]

        if search_counter >= 10:
            pulse_summation += pulse_single_trace
            baseline_summation += baseline
            avg_counter += 1


        search_timer = pulse_start_time + pulse_period
        search_counter += 1

        # print(pulse_start_time)

    pulse_avg = pulse_summation/avg_counter
    baseline_avg = baseline_summation/avg_counter

    return pulse_avg, baseline_avg, search_freqs

if __name__ == "__main__":
    plt.close('all')

    path   = os.path.join("/data/USRP_Noise_Scans",series.split("_")[0],series)
    os.chdir(path)

    objects = sorted(os.listdir(os.getcwd()))
    noise_files = []
    vna_files = []
    pulsing_files = []
    for fm in range(len(objects)):
        if objects[fm][-3:] == '.h5':
            if '1540' in objects[fm]:
                pulsing_files += [objects[fm][:]]
            elif 'USRP_VNA' in objects[fm]:
                vna_files += [objects[fm][:-3]]


    raw_f, raw_VNA, _ = puf.read_vna(vna_files[0]+'.h5')
    trim_f = raw_f[2000:-2000]
    trim_VNA = raw_VNA[2000:-2000]


    for pulse_file in pulsing_files:
        pulse_avg, baseline_avg, search_freqs = average_traces(pulse_file)



        for readout_idx in range(len(search_freqs)):
#            if readout_idx == pulse_idx:
#                label = '_pulse_'
#            else:
#                label = '_readout_'

            pulse_fig = plt.figure(pulse_file + '_' + str(search_freqs[readout_idx]))
            plt.plot(time,\
                     abs(pulse_avg[:,readout_idx]-baseline_avg[readout_idx]),\
                     color = c_wheel[readout_idx])
            plt.title(pulse_file + '_' + str(search_freqs[readout_idx]))


        # pulse_fig.add_axes([150, 0.1, 100])

        # abs_val_array = np.abs(trim_f-search_freqs[-1]*1e-6)
        # pulse_freq_idx = abs_val_array.argmin()
        #
        # plot_f = trim_f[pulse_freq_idx-10000:pulse_freq_idx+10000]
        # plot_VNA = trim_VNA[pulse_freq_idx-10000:pulse_freq_idx+10000]
        #
        # plt.figure(noise_file + '_pulse_on_VNA')
        # plt.plot(plot_f,20*np.log10(abs(plot_VNA)))
        # plt.plot(search_freqs[-1]*1e-6,\
        #          20*np.log10(abs(trim_VNA[pulse_freq_idx])),\
        #          color='r',\
        #          marker='.',\
        #          markersize=30)

        # pulse_fig.add_axes([])

        # readout_fig = plt.figure(noise_file + '_readout')
        # plt.plot(time,pulse_avg)
        # plt.title(noise_file + '_readout')




        color_idx = 0
        for search_freq in search_freqs:
            abs_val_array = np.abs(trim_f-search_freq*1e-6)
            pulse_freq_idx = abs_val_array.argmin()

            plot_f = trim_f[pulse_freq_idx-10000:pulse_freq_idx+10000]
            plot_VNA = trim_VNA[pulse_freq_idx-10000:pulse_freq_idx+10000]

            plt.figure(pulse_file + '_readout_on_VNA')
            plt.plot(plot_f,20*np.log10(abs(plot_VNA)),\
                     color = c_wheel[color_idx])
            plt.plot(search_freq*1e-6,\
                     20*np.log10(abs(trim_VNA[pulse_freq_idx])),\
                     color='r',\
                     marker='.',\
                     markersize=30)
            # plt.plot(trim_f,20*np.log10(abs(trim_VNA)))
            color_idx += 1






    # real_part = raw_noise.real
    # imag_part = raw_noise.imag
    # plt.figure('_S21')
    # plt.plot(real_part[1000::plot_decimation],imag_part[1000::plot_decimation],alpha = 0.5)


    plt.show()
