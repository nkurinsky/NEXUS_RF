## Import the relevant modules
import sys, os
import time, datetime
import argparse
import h5py
import numpy as np
from pathlib import Path
from copy import deepcopy

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
default_daq_params = {
    "rate"    : 100e6,
    "tx_gain" : 0,
    "rx_gain" : 17.0,
    "LO"      : [4.25e9],         ## [Hz] Round numbers, no finer than 50 MHz. Always must be list, same length as res freqs
}

## Set VNA sweep parameters
default_vna_params = {
    "f_span_kHz" : 200,        ## Symmetric about the center frequency
    "points"     : 2000,       ## Defined such that we look at 100 Hz windows
    "duration"   : 15,         ## [Sec] ## IF_BW ~= points / duration
}

## Set timestream parameters
default_readout_params = {
    "res"            : [4.241265],          ## Always must be list [GHz]
    "tracking_tones" : [4.231e9,4.251e9],   ## Cleaning tones to remove correlated noise [GHz]
    "powers"         : np.arange(start=-60,stop=-10,step=5),  ## dBm
    "duration"       : 15,                  ## [Sec] ## IF_BW ~= points / duration
    "cal_deltas"     : np.linspace(start=-0.05, stop=0.05, num=3), ## These deltas are fractions of the central frequency. This can be used to do a pseudo-VNA post facto
    "cal_lapse_sec"  : 10.,
}

default_params = {
    "daq": default_daq_params,
    "vna": default_vna_params,
    "ro" : default_readout_params,
}

## Where to save the output data (hdf5 files)
dataPath = Path('/data/USRP_Noise_Scans')

def parse_args():
    ## Instantiate the parser
    parser = argparse.ArgumentParser(description='Acquire a noise timestream with the USRP using the GPU_SDR backend.')

    ## Update the DAQ parameters
    parser.add_argument('--rate'  , '-r' , type=float, 
        default = default_daq_params["rate"]/1e6, 
        help = f'Sampling frequency (default {default_daq_params["rate"]/1e6} Msps)')
    parser.add_argument('--txgain', '-tx', type=float, 
        default = default_daq_params["tx_gain"], 
        help = f'Tx gain factor (default {default_daq_params["tx_gain"]})')
    parser.add_argument('--rxgain', '-rx', type=float,
        default = default_daq_params["rx_gain"], 
        help = f'Tx gain factor (default {default_daq_params["rx_gain"]})')
    parser.add_argument('--LOfrq' , '-f' , type=float, 
        default = default_daq_params["LO"]/1e6,
        help = f'LO frequency in MHz (default {LO/1e6} MHz)')

    ## Update the VNA parameters
    parser.add_argument('--VNAfspan', '-fv', type=float, 
        default = default_vna_params["f_span_kHz"],
        help = f'Frequency span in kHz over which to do the VNA scan (default {f_span_kHz} kHz)')
    parser.add_argument('--points'  , '-p' , type=int, 
        default = default_vna_params["points"], 
        help = f'Number of points used in the scan (default {default_vna_params["points"]} points)')
    parser.add_argument('--timeVNA' , '-Tv', type=float, 
        default = default_vna_params["duration"], 
        help = f'Duration of the VNA scan in seconds per iteration (default {default_vna_params["duration"]} seconds)')

    ## Update the Timestream Readout parameters
    parser.add_argument('--resFreq',   '-fr', type=float, nargs='+'
        default = default_readout_params["res"],
        help = f'List of resonant frequencies in GHz (default {default_readout_params["res"]} GHz)')
    parser.add_argument('--trkFreq',   '-ft', type=float,  nargs='+'
        default = default_readout_params["tracking_tones"],
        help = f'List of tracking frequencies in Hz (default {default_readout_params["tracking_tones"]} Hz)')
    parser.add_argument('--power',      '-P' , type=float, nargs='+'
        default = default_readout_params["powers"], 
        help = f'RF powers applied in dBm. (default {default_readout_params["powers"]} dBm)')
    parser.add_argument('--timeNoise', '-Tn', type=float, 
        default = default_readout_params["duration"], 
        help = f'Duration of the noise scan in seconds (default {default_readout_params["duration"]} seconds)')
    parser.add_argument('--calDeltas', '-cd', type=float, nargs='+'
        default = default_readout_params["cal_deltas"], 
        help = f'Fractional offset for calibration tones (default {default_readout_params["cal_deltas"]})')
    parser.add_argument('--calTime',   '-Tc', type=float, 
        default = default_readout_params["cal_lapse_sec"], 
        help = f'Duration of timestream for calibration tones (default {default_readout_params["cal_lapse_sec"]} sec)')
    
    ## Data path optional arguments
    parser.add_argument('--d', type=str, 
        default = dataPath,
        help = f'Top-level directory for storing VNA data (default: {dataPath})')

    ## Flag for debug mode
    parser.add_argument('--debug', action='store_true',
        help='Print scan settings without taking measurements')
    
    ## Now read the arguments
    args = parser.parse_args()

    ## Do some conditional checks

    ## Reject NaN and infinity in all scalar numeric arguments
    for name in ("r", "tx", "rx", "fv", "p", "Tv", "Tn", "Tc"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")






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

    ## Validate the data path
    desired_path = Path(args.d)

    params = {
        "daq": {
            "rate"    : args.rate,
            "tx_gain" : args.txgain,
            "rx_gain" : args.rxgain,
            "LO"      : 4.25e9, ## >TODO<
        },
        "vna": {
            "f_span_kHz" : args.VNAfspan,
            "points"     : args.points,
            "duration"   : args.timeVNA,
        },
        "ro" : {
            "res"            : [4.241265],  ## >TODO<
            "tracking_tones" : [4.231e9,4.251e9],  ## >TODO<
            "powers"         : np.arange(start=-60,stop=-10,step=5),  ## >TODO<
            "duration"       : args.timeNoise,
            "cal_deltas"     : np.linspace(start=-0.05, stop=0.05, num=3), ## >TODO<
            "cal_lapse_sec"  : args.calTime,
        }
    }

    return params, desired_path, args.debug


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

        daq.run_full_suite(series, daq_params, [res], run_type="Noise", h5_group_obj=fyle, subrun_id=i)

        ## Free up the file now that we're done
        fyle.close()

    ## Disconnect from the USRP server
    u.Disconnect()
