import sys
import os
sys.path.append('../GPU_SDR-master')
sys.path.append('../PyMKID-master')
sys.path.append('../PyMKID-master/GPIB_instruments')
import PyMKID_USRP_functions as puf
import numpy as np
import h5py
import sig_gen
import plot_pulse

pulse_filename = '201020OW200127p1_pulse_power_scan.h5'

sig_gen.on_off_switch(0)

os.system('wsl python take_VNAs.py')

os.system('python plot_VNAs.py')

latest_VNA = puf.get_latest_file('_VNA_','.h5')

readout_fr, readout_Qr = puf.vna_file_fit(latest_VNA,\
                                          [4.24013,4.2491],
                                          show=True)

num_scan_freqs = 21
steps = np.linspace(-2,2,num_scan_freqs)
print(steps)
idx = 0
scan_freq_array = np.zeros((num_scan_freqs,2))
scan_mean_array = np.zeros((num_scan_freqs,2),dtype=complex)
for step in steps:
    noise_scan_string = ' '
    for f, Q  in zip(readout_fr,readout_Qr):
        noise_scan_string += str((f + step*f/Q)*1e9) + ' '
    
    os.system('wsl python take_noise.py' + noise_scan_string)
    latest_noise = puf.get_latest_file('_Noise_','.h5')
    scan_freqs, scan_means = puf.avg_noi(latest_noise)
    
    scan_freq_array[idx,:] = scan_freqs
    scan_mean_array[idx,:] = scan_means
    
    os.remove(latest_noise)
    idx += 1

with h5py.File(pulse_filename,'a') as fyle:
    fr_scan_group = fyle.create_group("fr scan frequencies")
    fr_scan_group.create_dataset('scan frequencies',data = scan_freq_array)
    fr_scan_group.create_dataset('S21 of scan frequencies',data = scan_mean_array)

pulse_frequency = readout_fr[1]*1e9

sig_gen.set_frequency(pulse_frequency)

resonance_string = str(readout_fr*1e9)

resonance_string = resonance_string[1:-1]

sig_gen_powers = [-5, -10, -15]

sig_gen.on_off_switch(1)

for power in sig_gen_powers:
    sig_gen.set_power(power)

    os.system('wsl python take_noise.py ' + resonance_string)

    latest_noise = puf.get_latest_file('Noise','.h5')

    pulse_avg, baseline_avg, search_freqs = plot_pulse.average_traces(latest_noise)

    with h5py.File(pulse_filename,'a') as fyle:
        pulse_freq_group = fyle.create_group('pulse power = ' + str(power))
        pulse_freq_group.create_dataset("average pulse shape",data=pulse_avg)
        pulse_freq_group.create_dataset("average baseline level",data=baseline_avg)

sig_gen.on_off_switch(0)
