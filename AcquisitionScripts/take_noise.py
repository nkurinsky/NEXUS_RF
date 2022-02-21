from __future__ import division

## Import the relevant modules
import sys, os
import numpy as np
import time
import h5py

## Try to read in the USRP modules
## Exit out if you can't after adjusting path
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../DeviceControl/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")
        exit()

try:
    import PyMKID_USRP_functions as puf
except ImportError:
    try:
        sys.path.append('../AnalysisScripts')
        import PyMKID_USRP_functions as puf
    except ImportError:
        print("Cannot find the PyMKID_USRP_functions package")
        exit()

try:
    import PyMKID_USRP_import_functions as puf2
except ImportError:
    print("Cannot find the PyMKID_USRP_import_functions package")
    exit()

## Attempt to connect to GPU SDR server
if not u.Connect():
    u.print_error("Cannot find the GPU server!")
    
## Set some noise scan parameters
rate    = 100e6
tx_gain = 0
rx_gain = 17.5
LO      = 4.25e9 ## Nice round numbers, don't go finer than 50 MHz
res     = 4.242170 ## In GHz
tracking_tones = np.array([]) # np.array([4.235e9,4.255e9])

## Set the stimulus powers to loop over
powers = np.arange(start = -50,
                  stop  = -40,
                  step  =   5)
n_pwrs = len(powers)

## Set the deltas to scan over in calibrations
cal_deltas = [0 ]#np.linspace(start=-0.05, stop=0.05, num=3)
n_c_deltas = len(cal_deltas)

# powers = [-70.-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10]

## Create some output containers
## Each entry in these arrays is itself an array
freqs = np.zeros(n_pwrs, dtype=object)
means = np.zeros(n_pwrs, dtype=object)

## Loop over the powers considered
for i in np.arange(n_pwrs):
    ## Pick this power
    power = powers[i]

    ## Ensure the power doesn't go above -25 dBm
    if power > -25:
        USRP_power = -25
        tx_gain    = power - USRP_power
    else:
        USRP_power = power

    ## Calculate some derived quantities
    N_power = np.power(10.,(((-1*USRP_power)-14)/20.))
    pwr_clc = np.round(-14-20*np.log10(N_power),2)

    ## Print some diagnostic text
    print( pwr_clc,' dBm')
    print( N_power)

    ## Do a VNA run with the USRP
    vna_file, delay = puf2.vna_run(tx_gain        = tx_gain , 
                                   rx_gain        = rx_gain ,
                                   iter           = 1       ,
                                   rate           = rate    ,
                                   freq           = LO      ,
                                   front_end      = 'A'     ,
                                   f0             = -10e6   ,
                                   f1             = -5e6    ,
                                   lapse          = 10      ,
                                   points         = 1e5     ,
                                   ntones         = N_power ,
                                   delay_duration = 0.1     ,
                                   delay_over     = 'null'  )

    ## Fit the data acquired in this noise scan
    fs, qs, _,_,_,_,_ = puf.vna_file_fit(vna_file + '.h5',[res],show=False)
    
    ## Extract the important parameters from fit
    f = fs[0]*1e9
    q = qs[0]
    print(f,q)

    ## Create some output objects
    ## Each entry is a single number
    cal_freqs = np.zeros(n_c_deltas)
    cal_means = np.zeros(n_c_deltas)

    ## For each power, loop over all the calibration offsets
    for j in np.arange(n_c_deltas):
        ## Pick this calibration delta
        delta = cal_deltas[j]

        readout_tones  = (f + delta*float(f)/q) + tracking_tones
        n_ro_tones     = len(readout_tones)
        # readout_tones = [tracking_tone_1, tracking_tone_2]

        amplitudes     = 1./N_power * np.zeros(n_ro_tones)

        relative_tones = np.zeros(n_ro_tones)
        for k in np.arange(n_ro_tones):
            relative_tones[k] = float(readout_tones[k]) - LO

        ## Another parameter for the noise run
        duration = 10

        ## Do a noise run with the USRP
        noise_file = puf2.noise_run(rate       = rate       ,
                                    freq       = LO         ,
                                    front_end  = "A"        ,
                                    tones      = relative_tones,
                                    lapse      = duration   ,
                                    decimation = 100         ,
                                    tx_gain    = tx_gain    ,
                                    rx_gain    = rx_gain    ,
                                    vna        = None       ,
                                    mode       = "DIRECT"   ,
                                    pf         = 4          ,
                                    trigger    = None       ,
                                    amplitudes = amplitudes ,
                                    delay      = delay*1e9  )
        ## Add an extension to the file path
        noise_file += '.h5'

        time_threshold = duration / 2

        frequencies_scanned, noise_mean_scanned = puf.avg_noi(noise_file,time_threshold=time_threshold)

        ## Store the result in the internal output arrays
        cal_freqs[j] = frequencies_scanned[0]
        cal_means[j] = noise_mean_scanned[0]

        if delta != 0:
            os.remove(noise_file)

    ## Store the resulting array from the inner loop in the
    ## overall output containers (arrays of arrays)
    freqs[i] = cal_freqs
    means[i] = cal_means

## Create an output file
with h5py.File('noise_averages.h5','a') as fyle:
    fyle.create_dataset('freqs',data=freqs)
    fyle.create_dataset('means',data=means)

## Disconnect from the USRP server
u.Disconnect()
