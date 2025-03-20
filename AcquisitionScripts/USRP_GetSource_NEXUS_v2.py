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

## Set run parameters
run_type = "Background"
N_full_runs = 5      ## Number of "series" to take
run_time = 60 * 60.0 ## Time of each "series"; passed in seconds, default to one hour
sub_time =  5 * 60.0 ## Time of each "trace" comprising this series; passed in seconds, default to five minutes
nse_time =      30.0 ## Time of the noise acquisition ahead of all "traces"; passed in seconds, default to thirty seconds

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
vna_points = 2000       ## Defined such that we look at 100 Hz windows
vna_time   = 15         ## [Sec] ## IF_BW = points / duration

## Set the non-resonator tracking tones
# tracking_tones = np.array([5.3774e9,5.3974e9]) ## (Al or Nb 7)    In Hz a.k.a. cleaning tones to remove correlated noise
tracking_tones = np.array([4.231e9,4.251e9]) ## (Al or Nb 7)    In Hz a.k.a. cleaning tones to remove correlated noise
# tracking_tones = np.array([4.193e9,4.213e9]) ## (Nb 6)  In Hz a.k.a. cleaning tones to remove correlated noise

## Set the stimulus powers
power = -15

## Set the deltas to scan over in calibrations
## These deltas are fractions of the central frequency
## This can be used to do a pseudo-VNA post facto
cal_deltas = np.linspace(start=-0.05, stop=0.05, num=3)
cal_lapse_sec = 10.

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_'+run_type+'_Data'

def parse_args():
    ## Instantiate the parser
    parser = argparse.ArgumentParser(description='Acquire a noise timestream with the USRP using the GPU_SDR backend.')

    ## Read the arguments for stimulus and digitization parameters
    parser.add_argument('--power'    , '-P' , type=float, default = power, 
        help='RF power applied in dBm. (default '+str(power)+' dBm)')
    parser.add_argument('--txgain'   , '-tx', type=float, default = tx_gain, 
        help='Tx gain factor (default '+str(tx_gain)+')')
    parser.add_argument('--rxgain'   , '-rx', type=float, default = rx_gain, 
        help='Rx gain factor (default '+str(rx_gain)+')')
    parser.add_argument('--LOfrq' , '-f' , type=float, default=LO/1e6,
        help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default '+str(LO/1e6)+' MHz)')
    parser.add_argument('--rate'     , '-R' , type=float, default = rate/1e6, 
        help='Sampling frequency (default '+str(rate/1e6)+' Msps)')

    ## Read the arguments for duration parameters
    parser.add_argument('--timeVNA',   '-Tv' , type=float, default=vna_time, 
        help='Duration of the VNA scan in seconds per iteration (default '+str(vna_time)+' seconds)')
    parser.add_argument('--timeNoise', '-Tn' , type=float, default=nse_time, 
        help='Duration of the pre-run noise acqusition scan in seconds (default '+str(nse_time)+' seconds)')
    parser.add_argument('--timeSub',   '-Ts' , type=float, default=sub_time, 
        help='Duration of the sub-run acqusition scan in seconds (default '+str(sub_time)+' seconds)')
    parser.add_argument('--timeRun',   '-Tr' , type=float, default=run_time, 
        help='Duration of the full-run acqusition scan in seconds (default '+str(run_time)+' seconds)')
    parser.add_argument('--Nruns',     '-Nr' , type=int,   default=N_full_runs, 
        help='Total number of full-duration runs to take (default '+str(N_full_runs)+' runs)')

    ## Read the arguments for the VNA scan details
    parser.add_argument('--VNAfspan', '-fv', type=float, default=f_span_kHz,
        help='Frequency span in kHz over which to do the VNA scan (default '+str(f_span_kHz)+' kHz)')
    parser.add_argument('--points'   , '-p' , type=int  , default=vna_points, 
        help='Number of points used in the scan (default '+str(vna_points)+' points)')
    parser.add_argument('--iter'  , '-i' , type=int, default=1, 
        help='How many iterations to perform (default 1)')
    
    args = parser.parse_args()

    ## Do some conditional checks

    print("Power(s):", args.power, type(args.power))

    min_pwer = -70.0
    max_pwer = -15.0
    if (args.power < min_pwer):
        print("Power",args.power,"too Low! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to minimum...")
        args.power = min_pwer

    if (args.power > max_pwer):
        print("Power",args.power,"too High! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to maximum...")
        args.power = max_pwer

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

    ## Ensure the power doesn't go above -25 dBm
    ## Due to power splitting across tones
    this_power = args.power
    if this_power > -25:
        daq_params["rf_power"] = -25
        daq_params["tx_gain"]  = this_power - daq_params["rf_power"]
    else:
        daq_params["rf_power"] = this_power

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    ## Loop over each of the full runs/series requested
    for i in np.arange(args.Nruns):

        ## Define and create the output directories
        date_path, series_path, series = daq.get_paths(dataPath)
        os.chdir(series_path)

        ## Instantiate an output file
        fyle = h5py.File('noise_averages_'+series+'.h5','w')

        ## Update the run parameters to ensure we do calibrations
        daq_params["stream"]["cal_deltas"] = cal_deltas
        daq_params["stream"]["duration_s"] = args.timeNoise

        ## Now do a line delay, VNA, and noise acquisition
        daq.run_full_suite(series, daq_params, [res], run_type="Noise", h5_group_obj=fyle, subrun_id=i)

        ## Update the run parameters so that we don't do calibrations
        daq_params["stream"]["cal_deltas"] = [0.0]
        daq_params["stream"]["duration_s"] = args.timeSub

        ## Start a timer to keep track of when we hit the end of a full run
        total_acq_time = 0
        subrun_cntr = 0

        ## Loop over the subruns to get desired total acqusition duration for this series
        while total_acq_time < args.timeRun:

            ## Create an H5 group for this trace in the summary file
            gTrace = fyle.create_group('Trace'+str(int(subrun_cntr)))
            gTrace.attrs.create("timestart", time.time())
            gTrace.attrs.create("duration", args.timeSub)

            ## Take a "trace" for the subRun time ; note that we don't care about the output since we aren't saving the calibration tones
            daq.run_stream(series, daq_params, h5_group_obj=None, cooltime_s=0, run_type=run_type, suffix=subrun_cntr)

            ## Update for next iteration of the loop
            total_acq_time += args.timeSub
            subrun_cntr += 1

        ## Write a file to indicate this Series has completed
        with open('acq-complete', 'w') as the_file:
            the_file.write(str(time.time())+'\n')

        ## Free up the file now that we're done
        fyle.close()

    ## Disconnect from the USRP server
    u.Disconnect()
