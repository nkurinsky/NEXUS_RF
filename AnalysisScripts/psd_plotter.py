import sys, os, glob, h5py
import time, datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../BackendTools')
import PyMKID_USRP_functions as PUf
import PyMKID_resolution_functions as Prf

series    = '20220407_153101'
# data_path = os.path.join('/home/dylan/KID_data/',series)
data_path = os.path.join('/data/USRP_Noise_Scans',series.split('_')[0],series)

## Grab the file with no calibration offset
file_list = np.sort(glob.glob(data_path+"/*.h5"))

summary_file = "None"
delay_file   = "None"
VNA_file     = "None"
tone_files   = []

for file in file_list:
    
    if "Delay" in file:
        delay_file = file
        
    if "VNA" in file:
        VNA_file = file
        
    if "noise_averages" in file:
        summary_file = file
        
    if ("delta" in file)  and not ("cleaned" in file):
        tone_files = np.append(tone_files, file)

print("Line Delay file: ",delay_file)
print("VNA scan file:   ",VNA_file)
print("Timestream files:",tone_files)
print("Summary file:    ",summary_file)

## First pull metadata from the noise averages summary file
fsum = h5py.File(summary_file, 'r')
md   = fsum['Power0']
print(md.keys())
print(md.attrs.keys())

LOfreq   = md.attrs["LOfreq"]
N_power  = md.attrs["N_power"]
delay_ns = md.attrs["delay_ns"]
power    = md.attrs["power"]
rate     = md.attrs["rate"]
rx_gain  = md.attrs["rx_gain"]
tx_gain  = md.attrs["tx_gain"]

frequencies = np.array(md['freqs'])
noise_means = np.array(md['means'])

S0 = md['Scan0'] ; print(S0.keys())
# S1 = md['Scan1'] ; print(S1.keys())
# S2 = md['Scan2'] ; print(S2.keys())
print(md['VNA'])
print(frequencies)
print(noise_means)

print("Scan0 tones:", np.array(S0['readout_tones']))
# print("Scan1 tones:", np.array(S1['readout_tones']))
# print("Scan2 tones:", np.array(S2['readout_tones']))

## Now look at on-resonance, no cal delta file
PSD_lo_f = int(1e2)  ## chunk up to [Hz]
PSD_hi_f = int(5e4)  ## decimate down to  [Hz]

print("Power: ",power)

_, noise_info = PUf.unavg_noi(tone_files[0])
noise_total_time = noise_info['time'][-1]
noise_fs = 1./noise_info['sampling period']
noise_readout_f = noise_info['search freqs'][0]

num_chunks = int(noise_total_time*PSD_lo_f)
noise_decimation = int(noise_fs/PSD_hi_f)

print("Will separate data into ", num_chunks      , "chunks to achieve the requested", "{:.2e}".format(PSD_lo_f),' Hz low  end of the PSD')
print("Additional decimation by", noise_decimation, "needed to achieve the requested", "{:.2e}".format(PSD_hi_f),' Hz high end of the PSD')

powers, PSDs, res, timestreams = Prf.PSDs_and_cleaning(tone_files[0], VNA_file,
                                                      extra_dec  = noise_decimation,
                                                      num_chunks = num_chunks,
                                                      blank_chunks = int(0.3*num_chunks),
                                                      removal_decimation = 1)

plt.show()

# ## Look for pulses
# f = h5py.File(tone_files[0], 'r')
# f_clean = h5py.File(tone_files[0].split('.')[0]+"_cleaned.h5", 'r')
# # print("Top-level:", f.keys())
# # print("Top-level:", f_clean.keys())
# # print("Cleaned data:", f_clean['cleaned_data'])
# # print("Radius:", f_clean['radius'])
# # clean_data = f_clean['cleaned_data']
# # res_S21_clean = np.array(clean_data[:,0])

# # plt.figure()
# # plt.plot(logMag(res_S21_clean))

# pulses = np.array(f['pulses'])
# print("Pulses:   ", pulses)

# # for i in np.arange(len(pulses)):
# #     t_win_min = pulses[i]-0.003
# #     t_wim_max = pulses[i]+0.010
    
# #     timestreams['radius simple clean']

# plt.plot(timestreams['arc simple clean'][:,0],timestreams['arc simple clean'][:,1])
# plt.show()

# f.close()
# f_clean.close()
