## Import the relevant modules
import sys, os
import time, datetime
import argparse
import numpy as np

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
        sys.path.append('../BackendTools')
        import PyMKID_USRP_functions as puf
    except ImportError:
        print("Cannot find the PyMKID_USRP_functions package")
        exit()

try:
    import PyMKID_USRP_import_functions as puif
except ImportError:
    try:
        sys.path.append('../BackendTools')
        import PyMKID_USRP_import_functions as puif
    except ImportError:
        print("Cannot find the PyMKID_USRP_import_functions package")
    exit()

## Set DAQ parameters
rate    = 100e6
tx_gain = 0
rx_gain = 17.0
LO      = 4.25e9       ## (Al and Nb 7) [Hz] Round numbers, no finer than 50 MHz
# LO      = 4.20e9       ## (Nb 6) [Hz] Round numbers, no finer than 50 MHz
led_dec   = 100        ## Default decimation for the LED timestreams

## Set Resonator parameters
res     = 4.24198300      ## Al   [GHz]
# res     = 4.244760      ## Nb 7 [GHz]
# res     = 4.202830      ## Nb 6 [GHz]

## Set some VNA sweep parameters
f_span_kHz    = 140        ## Symmetric about the center frequency
points        = 1400       ## Defined such that we look at 100 Hz windows
vna_lapse_sec = 10         ## [Sec]

## Set the non-resonator tracking tones
tracking_tones = np.array([4.235e9,4.255e9]) ## (Al)    In Hz a.k.a. cleaning tones to remove correlated noise

## Set the stimulus power and noise timestream duration
power         = -40        ## [dBm]
nse_lapse_sec =  10        ## [Sec]

## Set the deltas to scan over in calibrations
## These deltas are fractions of the central frequency
## This can be used to do a pseudo-VNA post facto
cal_deltas = np.linspace(start=-0.05, stop=0.05, num=3)
n_c_deltas = len(cal_deltas)
cal_lapse_sec = 10.

## Set the number of minutes to acquire science data
sci_lapse_min = 1          ## [Min]

## File handling options
filename=None

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_Science_Runs/'

## Sub directory definitions
dateStr   = '' # str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath = '' # os.path.join(dataPath,dateStr)

series     = '' # str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = '' # os.path.join(sweepPath,series)

def get_paths():
    ## Sub directory definitions
    dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
    sweepPath = os.path.join(dataPath,dateStr)

    series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    seriesPath = os.path.join(sweepPath,series)

    return dateStr, sweepPath, series, seriesPath

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Acquire a noise timestream with the USRP using the GPU_SDR backend.')

    parser.add_argument('--power'    , '-P' , type=float, default = power, 
        help='RF power applied in dBm. (default '+str(power)+' dBm)')
    parser.add_argument('--txgain'   , '-tx', type=float, default = tx_gain, 
        help='Tx gain factor (default '+str(tx_gain)+')')
    parser.add_argument('--rxgain'   , '-rx', type=float, default = rx_gain, 
        help='Rx gain factor (default '+str(rx_gain)+')')
    parser.add_argument('--rate'     , '-R' , type=float, default = rate/1e6, 
        help='Sampling frequency (default '+str(rate/1e6)+' Msps)')
    parser.add_argument('--points'   , '-p' , type=int  , default=points, 
        help='Number of points use d in the scan (default '+str(points)+' points)')
    parser.add_argument('--timeVNA'  , '-Tv' , type=float, default=vna_lapse_sec, 
        help='Duration of the VNA scan in seconds per iteration (default '+str(vna_lapse_sec)+' seconds)')
    parser.add_argument('--timeNoise', '-Tn' , type=float, default=nse_lapse_sec, 
        help='Duration of the noise scan in seconds (default '+str(nse_lapse_sec)+' seconds)')
    parser.add_argument('--timeScience', '-Ts' , type=float, default=sci_lapse_min, 
        help='Duration of the noise scan in minutes (default '+str(sci_lapse_min)+' minutes)')

    args = parser.parse_args()

    ## Do some conditional checks

    if (args.power is not None):
        print("Power(s):", args.power, type(args.power))

        power = args.power

        min_pwer = -70.0
        max_pwer = -15.0

        if (power < min_pwer):
            print("Power",args.power,"too Low! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to minimum...")
            power = min_pwer

        if (power > max_pwer):
            print("Power",args.power,"too High! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to maximum...")
            power = max_pwer

    if (args.rate is not None):
        args.rate = args.rate * 1e6 ## Store it as sps not Msps
        if (args.rate > rate):
            print("Rate",args.rate,"is too High! Optimal performance is at",rate,"samples per second")
            args.rate = rate

    if (args.iter is not None):
        if (args.iter < 0):
            args.iter = 1

    return args

def create_dirs():
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    if not os.path.exists(sweepPath):
        os.makedirs(sweepPath)

    if not os.path.exists(seriesPath):
        os.makedirs(seriesPath)
    print ("Scan stored as series "+series+" in path "+sweepPath)
    return 0

def runScience(N_acqs, tx_gain, rx_gain, _iter, rate, freq, front_end, fspan, lapse_VNA, lapse_noise, points, ntones, delay_duration, h5_group_obj=None):
    
    delay = puif.run_delay(series=series, 
        tx_gain      = tx_gain, 
        rx_gain      = rx_gain, 
        rate         = rate, 
        freq         = freq, 
        front_end    = front_end, 
        lapse_delay  = delay_duration, 
        delay_over   = None, 
        h5_group_obj = h5_group_obj)

    f, q = puif.run_vna(series=series, 
        res          = res, 
        tx_gain      = tx_gain, 
        rx_gain      = rx_gain,  
        _iter        = _iter, 
        rate         = rate, 
        freq         = freq, 
        front_end    = front_end,  
        fspan        = fspan, 
        lapse_VNA    = lapse_VNA, 
        points       = points, 
        ntones       = ntones, 
        h5_group_obj = h5_group_obj)

    cal_freqs, cal_means = puif.run_noise(series=series,
        delay        = delay, 
        f            = f, 
        q            = q, 
        cal_deltas   = cal_deltas, 
        tracking_tones = tracking_tones, 
        tx_gain      = tx_gain, 
        rx_gain      = rx_gain,  
        rate         = rate, 
        freq         = freq, 
        front_end    = front_end,  
        lapse_noise  = lapse_noise,
        cal_lapse_sec= cal_lapse_sec, 
        points       = points, 
        ntones       = ntones, 
        h5_group_obj = h5_group_obj,
        idx          = None)

    ## We'll need this for when we do subsequent science acquisitions
    cal_fs_obj = np.zeros(shape=(2,len(cal_freqs))                   ) ; cal_fs_obj[1] = cal_freqs
    cal_ms_obj = np.zeros(shape=(2,len(cal_means)),dtype='complex128') ; cal_ms_obj[1] = cal_means

    ## Keep doing noise runs until we reach the 
    for i in 1+np.arange(N_acqs-1):

        cal_freqs2, cal_means2 = puif.run_noise(series=series,
            delay        = delay, 
            f            = f, 
            q            = q, 
            cal_deltas   = cal_deltas, 
            tracking_tones = tracking_tones, 
            tx_gain      = tx_gain, 
            rx_gain      = rx_gain,  
            rate         = rate, 
            freq         = freq, 
            front_end    = front_end,  
            lapse_noise  = lapse_noise,
            cal_lapse_sec= cal_lapse_sec, 
            points       = points, 
            ntones       = ntones, 
            h5_group_obj = h5_group_obj,
            idx          = i)

        cal_fs_obj = np.r_[cal_fs_obj, [cal_freqs2]]
        cal_ms_obj = np.r_[cal_ms_obj, [cal_means2]]

    return cal_fs_obj, cal_ms_obj

def doRun(this_power):

    ## Create the output directories
    create_dirs()
    os.chdir(seriesPath) ## When doing this, no need to provide subfolder

    ## Instantiate an output file
    fyle = h5py.File(os.path.join(seriesPath,'noise_averages_'+series+'.h5'),'w')

    ## Ensure the power doesn't go above -25 dBm
    ## Due to power splitting across tones
    if this_power > -25:
        USRP_power   = -25
        args.txgain = this_power - USRP_power
    else:
        USRP_power   = this_power

    ## Calculate some derived quantities
    N_power = np.power(10.,(((-1*USRP_power)-14)/20.))
    pwr_clc = np.round(-14-20*np.log10(N_power),2)

    print("Initializing Noise Scan...")
    print(pwr_clc, 'dBm of power')
    print(args.txgain, 'dB of additional gain')
    print(N_power, 'is the equivalent number of tones needed to split the DAQ power into the above amount')

    ## Create an h5 group for this data, store some general metadata
    gPower = fyle.create_group('Power'+str(0)) #+str(i))
    gPower.attrs.create("power",   USRP_power)
    gPower.attrs.create("tx_gain", args.txgain)
    gPower.attrs.create("rx_gain", args.rxgain)
    gPower.attrs.create("N_power", N_power)
    gPower.attrs.create("rate",    args.rate)
    gPower.attrs.create("LOfreq",  args.LOfrq)

    ## Determine how many noise acquisitions to run
    N_acqs = int( np.max([ 1. , args.timeScience / (args.timeNoise/60.) ]) )
    gPower.attrs.create("N_acqs",  N_acqs)
    print("Running "+str(N_acqs)+" acquisition(s) of "+str(args.timeNoise)+" seconds each")

    cal_freqs, cal_means = runScience(
        N_acqs  = N_acqs,
        tx_gain = args.txgain,
        rx_gain = args.rxgain,
        _iter   = args.iter,
        rate    = args.rate,        ## Passed in Samps/sec
        freq    = args.LOfrq,       ## Passed in Hz
        front_end = "A",
        fspan   = args.VNAfspan,
        lapse_VNA   = args.timeVNA,   ## Passed in seconds
        lapse_noise = args.timeNoise, ## Passed in seconds
        lapse_laser = args.timeLaser, ## Passed in seconds
        laser_dec   = args.ledDec,
        points  = args.points,
        ntones  = N_power,
        delay_duration = 0.1, # args.delay_duration,
        h5_group_obj = gPower) #args.delay_over)

    ## Store the resulting arrays in this h5 group
    gPower.create_dataset('freqs',data=cal_freqs)
    gPower.create_dataset('means',data=cal_means)

    ## Close h5 file for writing
    fyle.close()

if __name__ == "__main__":

    
    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    ## Run the science acquisition for the indicated power
    dateStr, sweepPath, series, seriesPath = get_paths()
    doRun(power)

    ## Disconnect from the USRP server
    u.Disconnect()
