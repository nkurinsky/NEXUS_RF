from __future__ import division
import sys,os
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../../Devices/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

import numpy as np
import time
import PyMKID_USRP_functions as puf
import PyMKID_USRP_import_functions as puf2
import h5py

if not u.Connect():
    u.print_error("Cannot find the GPU server!")
    exit()


rate = 200e6
tx_gain = 0
rx_gain = 17.5
LO = 4.2e9
res = 4.244905
tracking_tones = [4.235e9,4.255e9]

powers = [-70.-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10]
# powers = [-10]
calibration_deltas = [-0.05, 0, 0.05]

frequencies = []
means = []


for power in powers:

    if power > -25:
        USRP_power = -25
        tx_gain = power - USRP_power
    else:
        USRP_power = power

    N_power = 10**(((-1*USRP_power)-14)/20)
    print( round(-14-20*np.log10(N_power),2),' dBm')
    print( N_power)


    vna_file, delay = puf2.vna_run(tx_gain=tx_gain, \
                                   rx_gain = rx_gain,\
                                   iter=1,\
                                   rate=rate,
                                   freq=LO,\
                                   front_end='A',\
                                   f0=43e6,f1=48e6,\
                                   lapse=20,\
                                   points=1e5,\
                                   ntones=N_power,\
                                   delay_duration=0.1,\
                                   delay_over='null')

    fs, qs, _,_,_,_,_ = puf.vna_file_fit(vna_file + '.h5',[res],show=False)

    f = fs[0]*1e9
    q = qs[0]

    cal_iter = 0
    calibration_frequencies = []
    calibration_means = []
    for delta in calibration_deltas:
        readout_tones = [f + delta*float(f)/q] + tracking_tones
        # readout_tones = [tracking_tone_1, tracking_tone_2]

        amplitudes = [1./N_power for x in readout_tones]


        relative_tones = []
        for readout_tone in readout_tones:
            relative_tone = float(readout_tone) - LO
            relative_tones.append(relative_tone)



##        noise_file = puf2.noise_run(rate=200e6,\
##                                    freq=LO,\
##                                    front_end="A",\
##                                    tones=np.asarray(relative_tones),\
##                                    lapse=1,\
##                                    decimation=40,\
##                                    tx_gain=tx_gain,\
##                                    rx_gain=rx_gain,\
##                                    vna=None,\
##                                    mode="DIRECT",\
##                                    pf=4,\
##                                    trigger=None,\
##                                    amplitudes=amplitudes,\
##                                    delay=delay*1e9)
##
##        _,transmission_means = puf.avg_noi(noise_file+'.h5',time_threshold=0.5)
##        res_transmission = abs(transmission_means[0])
##        tracking_transmission = abs(transmission_means[1])
##        res_tracking_ratio = res_transmission / tracking_transmission
##        amplitudes = [1./N_power, 1./N_power*res_tracking_ratio]
##        os.remove(noise_file + '.h5')

        if delta != 0:
            duration = 50
        else:
            duration = 50

        noise_file = puf2.noise_run(rate=200e6,\
                                    freq=LO,\
                                    front_end="A",\
                                    tones=np.asarray(relative_tones),\
                                    lapse=duration,\
                                    decimation=40,\
                                    tx_gain=tx_gain,\
                                    rx_gain=rx_gain,\
                                    vna=None,\
                                    mode="DIRECT",\
                                    pf=4,\
                                    trigger=None,\
                                    amplitudes=amplitudes,\
                                    delay=delay*1e9)
        noise_file += '.h5'

        time_threshold = duration / 2


        frequencies_scanned, noise_mean_scanned = puf.avg_noi(noise_file,time_threshold=time_threshold)
        frequency = frequencies_scanned[0]
        noise_mean = noise_mean_scanned[0]

        calibration_frequencies.append(frequency)
        calibration_means.append(noise_mean)

        if delta != 0:
            os.remove(noise_file)

        cal_iter += 1

    frequencies.append(calibration_frequencies)
    means.append(calibration_means)

with h5py.File('noise_averages.h5','a') as fyle:
    fyle.create_dataset('frequencies',data = np.asarray(frequencies))
    fyle.create_dataset('means',data=np.asarray(means))


u.Disconnect()
