## Import the relevant modules
import sys, os
import time, datetime
import argparse
import h5py
import numpy as np

import USRP_Acquisition_Methods as daq

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../DeviceControl/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")
        exit()


## Set DAQ parameters
rate    = 100e6
tx_gain = 0
rx_gain = 17.0
# LO      = 5.35e9       ## (Al and Nb 7) [Hz] Round numbers, no finer than 50 MHz
LO      = 4.25e9       ## (Al and Nb 7) [Hz] Round numbers, no finer than 50 MHz
# LO      = 4.20e9       ## (Nb 6) [Hz] Round numbers, no finer than 50 MHz

## Set Resonator parameters
# res     = 5.38740       ## Al   [GHz]
res     = 4.241265      ## Al   [GHz] - FNAL-I, NR-24
# res     = 4.241665      ## Al   [GHz] - FNAL-I, NR-23
# res     = 4.241958      ## Al   [GHz]
# res     = 4.244553      ## Nb 7 [GHz]
# res     = 4.202830      ## Nb 6 [GHz]

## Set some VNA sweep parameters
f_span_kHz = 200        ## Symmetric about the center frequency
points     = 2000       ## Defined such that we look at 100 Hz windows
duration   = 15         ## [Sec] ## IF_BW = points / duration

## Set the non-resonator tracking tones
# tracking_tones = np.array([5.3774e9,5.3974e9]) ## (Al or Nb 7)    In Hz a.k.a. cleaning tones to remove correlated noise
tracking_tones = np.array([4.231e9,4.251e9]) ## (Al or Nb 7)    In Hz a.k.a. cleaning tones to remove correlated noise
# tracking_tones = np.array([4.193e9,4.213e9]) ## (Nb 6)  In Hz a.k.a. cleaning tones to remove correlated noise

## Set the stimulus powers to loop over
powers = np.array([-15])
# powers = np.array([-60,-55,-50,-45,-40,-35,-30])
# powers = np.arange(start=-60,stop=-10,step=5)
# powers = np.array([-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15])
# powers = np.array([-20,-15])

## Set the deltas to scan over in calibrations
## These deltas are fractions of the central frequency
## This can be used to do a pseudo-VNA post facto
cal_deltas = np.linspace(start=-0.05, stop=0.05, num=3)
cal_lapse_sec = 10.

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_Noise_Scans'


def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Acquire a noise timestream with the USRP using the GPU_SDR backend.')

    parser.add_argument('--power'    , '-P' , type=float, default = powers[0], 
        help='RF power applied in dBm. (default '+str(powers[0])+' dBm)')
    parser.add_argument('--txgain'   , '-tx', type=float, default = tx_gain, 
        help='Tx gain factor (default '+str(tx_gain)+')')
    parser.add_argument('--rxgain'   , '-rx', type=float, default = rx_gain, 
        help='Rx gain factor (default '+str(rx_gain)+')')
    parser.add_argument('--rate'     , '-R' , type=float, default = rate/1e6, 
        help='Sampling frequency (default '+str(rate/1e6)+' Msps)')
    parser.add_argument('--points'   , '-p' , type=int  , default=points, 
        help='Number of points used in the scan (default '+str(points)+' points)')
    parser.add_argument('--timeVNA'  , '-Tv' , type=float, default=duration, 
        help='Duration of the VNA scan in seconds per iteration (default '+str(duration)+' seconds)')
    parser.add_argument('--timeNoise', '-Tn' , type=float, default=duration, 
        help='Duration of the noise scan in seconds (default '+str(duration)+' seconds)')

    parser.add_argument('--iter'  , '-i' , type=int, default=1, 
        help='How many iterations to perform (default 1)')
    
    parser.add_argument('--LOfrq' , '-f' , type=float, default=LO/1e6,
        help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default '+str(LO/1e6)+' MHz)')
    parser.add_argument('--VNAfspan', '-fv', type=float, default=f_span_kHz,
        help='Frequency span in kHz over which to do the VNA scan (default '+str(f_span_kHz)+' kHz)')
    # parser.add_argument('--f0'    , '-f0', type=float, default=f0/1e6, 
    #     help='Baseband start frequency in MHz, absolute (default '+str(f0/1e6)+' MHz)')
    # parser.add_argument('--f1'    , '-f1', type=float, default=f1/1e6, 
    #     help='Baseband end frequency in MHz, absolute (default '+str(f1/1e6)+' MHz)')

    args = parser.parse_args()

    ## Do some conditional checks

    if (args.power is not None):
        print("Power(s):", args.power, type(args.power))

        powers[0] = args.power
        n_pwrs = len(powers)

        min_pwer = -70.0
        max_pwer = -15.0
        for i in np.arange(n_pwrs):
            if (powers[i] < min_pwer):
                print("Power",args.power,"too Low! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to minimum...")
                powers[i] = min_pwer

            if (powers[i] > max_pwer):
                print("Power",args.power,"too High! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to maximum...")
                powers[i] = max_pwer

    if (args.rate is not None):
        args.rate = args.rate * 1e6 ## Store it as sps not Msps
        if (args.rate > rate):
            print("Rate",args.rate,"is too High! Optimal performance is at",rate,"samples per second")
            args.rate = rate

    if (args.iter is not None):
        if (args.iter < 1):
            args.iter = 1

    ## MHz frequencies to Hz
    if (args.LOfrq is not None):
        args.LOfrq = args.LOfrq*1e6 ## Store it as Hz not MHz
    if (args.VNAfspan is not None):
        args.VNAfspan = args.VNAfspan*1e3 ## Store it as Hz not kHz
        if(args.VNAfspan > 1e7):
            print("Frequency range (",args.VNAfspan,") too large! Exiting...")
            exit(1)

    if(args.LOfrq is not None):
        if(np.any(np.array(args.LOfrq) > 6e9)):
            print("Invalid LO Frequency:",args.freq," is too High! Exiting...")
            exit(1)

    return args


if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Get the run parameter dictionary
    daq_params = daq.generate_daq_params(
        front_end="A", 
        rate=args.rate, 
        tx_gain=args.txgain, 
        rx_gain=args.rxgain, 
        LO_freq=args.LOfrq, 
        delay_duration=10.0, 
        vna_duration=args.timeVNA, 
        stream_duration=args.timeNoise, 
        calibration_duration=cal_lapse_sec,
        f_center_Hz=res*1e9, 
        f_span_Hz=args.VNAfspan,
        vna_npoints=args.points, 
        vna_iterations=args.iter, 
        cal_deltas=cal_deltas,
        tracking_tones=tracking_tones
    )

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    ## Loop over the powers considered
    for i in np.arange(len(powers)):

        ## Define and create the output directories
        date_path, series_path, series = daq.get_paths(dataPath)
        os.chdir(series_path) ## When doing this, no need to provide subfolder

        ## Instantiate an output file
        fyle = h5py.File('noise_averages_'+series+'.h5','w')

        ## Ensure the power doesn't go above -25 dBm
        ## Due to power splitting across tones
        this_power = powers[i]
        if this_power > -25:
            daq_params["rf_power"] = -25
            daq_params["tx_gain"]  = this_power - daq_params["rf_power"]
        else:
            daq_params["rf_power"] = this_power

        daq.run_full_suite(series, daq_params, res, type="Noise", h5_group_obj=fyle, subrun_id=i)

        ## Free up the file now that we're done
        fyle.close()

    ## Disconnect from the USRP server
    u.Disconnect()
