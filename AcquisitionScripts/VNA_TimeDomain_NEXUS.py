import sys, os
from time import sleep
import numpy as np
import datetime
import argparse

## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")
from VNAfunctions import *  #using the VNA to do a power sweep
from NEXUSFunctions import * #control NEXUS fridge

sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/BackendTools")
from VNAMeas import * #vna measurement class

## Parameters of the time domain acquisition
P_ctr  = -27.0          ## RF stimulus power [dBm]
f_res  = 4.24217938e9   ## Resonator central frequency [Hz]
lapse  = 30             ## Duration of acquisition [sec]
srate  = 100            ## Sampling rate [ksps]
npts   = 200000         ## Number of samples per trace

## Inherited parameters
bdwt   = srate * 1e3   ## IF Bandwith [sampling rate Hz]

## Where to save the output data (hdf5 files)
dataPath = '/data/VNATimeDomain/'

## Sub directory definitions
dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath = dataPath + '/' + dateStr

series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Script to take a frequency sweep with the CMT VNA at a set of specified RF powers')

    # Basic parameters of time domain acq
    parser.add_argument('--freq' , '-f', type=float, default=f_res,
                        help='LO frequency [Hz]')
    parser.add_argument('--power', '-p', type=float, default=P_ctr,
                        help='Minimum power for scan [dBm]')
    parser.add_argument('--time' , '-t', type=float, default=lapse,
                        help='Duration of the scan [seconds]')
    parser.add_argument('--rate' , '-r', type=float, default=srate,
                        help='Sampling frequency [ksps]')
    parser.add_argument('--npts' , '-N', type=int  , default=npts,
                        help='Number of points to acquire per single trace')

    # Data path optional arguments
    parser.add_argument('--directory', '-d', type=str, default=sweepPath,
                        help='Top-level directory for storing VNA data')

    # Now read the arguments
    args = parser.parse_args()

    # Do some conditional checks
    if (args.power is not None):
        if (args.power > -10):
            print(args.P1, "dBm is too large, setting power to -10 dBm")
            args.power = -10.0

    if (args.rate is not None):
        if (args.rate > 100.):
            print(args.rate, "ksps is too large, setting rate to 100 ksps")
            args.rate = 100.0

    if (args.npts is not None):
        if (args.npts > 200001):
            print(npts, "is too many samples, maxing out at 200001 samples (adjust time argument)")
            args.npts = 200001

    return args

def create_dirs():
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    if not os.path.exists(sweepPath):
        os.makedirs(sweepPath)

    print ("Scan stored as series "+series+" in path "+sweepPath)
    return 0

def run_scan():

    ## Diagnostic text
    print("Current Fridge MC Temperature (mK): ", nf1.getTemp()*1e3)

    print("--Time Domain Settings-------")
    print("-      RF Power (dB):", P_ctr)
    print("-   Resonator F (Hz):", f_res)
    print("-     Duration (sec):", lapse)
    print("- Sample rate (Ksps):", srate) ## Max 100 KHz
    print("-       IF Bandwidth:", bdwt)  ## Max 100 KHz
    print("-           N points:", npts)  ## Max 200001 samples

    ## Create a filename for this sweep
    output_filename = sweepPath +"/TimeSer_P"+str(P_ctr)+"_"+series

    ## Create a class to contain the sweep result
    sweep = VNAMeas(dateStr, series)
    sweep.power   = power
    sweep.n_avgs  = 0
    sweep.n_samps = npts
    sweep.f_min   = -1.0
    sweep.f_max   = f_res

    ## Grab and save the fridge temperature before starting sweep
    # sweep.start_T = np.array([nf1.getTemp(), nf2.getTemp()])
    sweep.start_T = np.array([ nf1.getTemp() ])

    ## Set the VNA stimulus power and take a frequency sweep
    v.setPower(P_ctr)
    times, S21_real, S21_imag = v.timeDomain(lapse, f_res, npts, ifb=bdwt)

    ## Grab and save the fridge temperature after sweep
    sweep.final_T = np.array([ nf1.getTemp() ])

    ## Save the result to our class instance
    sweep.frequencies = np.array(times)
    sweep.S21realvals = np.array(S21_real)
    sweep.S21imagvals = np.array(S21_imag)

    # Write our class to a file (h5)
    print("Storing data at:", sweep.save_hdf5(output_filename))

    ## Store the data in our file name (csv)
    # v.storeData(times, S21_real, S21_imag, output_filename)

    ## Diagnostic text
    print("Current Fridge MC Temperature (mK): ", nf1.getTemp()*1e3)
    print("Power scan complete.")
    return 0

if __name__ == "__main__":
    ## Initialize the NEXUS temperature servers
    nf1 = NEXUSHeater(server_ip="192.168.0.34",server_port=11034)

    ## Initialize the VNA
    v = VNA()

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Parameters of the power sweep (in dB)
    P_ctr  = args.power if args.power is not None else P_ctr
    f_res  = args.freq  if args.freq  is not None else f_res
    lapse  = args.time  if args.time  is not None else lapse
    srate  = args.rate  if args.rate  is not None else srate
    npts   = args.npts  if args.npts  is not None else npts

    ## Recalculate inherited params
    bdwt   = srate * 1e3   ## IF Bandwith [sampling rate Hz]
    
    ## Where to save the output data (hdf5 files)
    sweepPath = args.directory if args.directory is not None else sweepPath

    ## Create the output directories
    create_dirs()

    ## Run the time scan
    run_scan()