import sys, os
from time import sleep
import numpy as np
import datetime
import argparse

## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")

from VNAfunctions import *  #using the VNA to do a power sweep
from NEXUSFunctions import * #control NEXUS fridge
from VNAMeas import * #vna measurement class

## Parameters of the time domain acquisition
P_ctr  = -40.0      ## RF stimulus power [dBm]
f_res  = 4.24217e9  ## Resonator central frequency [Hz]
lapse  = 100        ## Duration of acquisition [sec]
srate  = 100        ## Sampling rate [Msps]

## Inherited parameters
bdwt   = srate * 1e6   ## IF Bandwith [sampling rate Hz]
npts   = bdwt * lapse ## N points to take

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
                        help='Sampling frequency [Msps]')

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
    print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
    print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())

    print("--Time Domain Settings-------")
    print("-      RF Power (dB):", P_ctr)
    print("-   Resonator F (Hz):", f_res)
    print("-     Duration (sec):", lapse)
    print("- Sample rate (Msps):", srate)
    print("-       IF Bandwidth:", bdwt)
    print("-           N points:", npts)

    ## Create a filename for this sweep
    output_filename = seriesPath +"/TimeSer_P"+str(P_ctr)+"_"+series

    ## Grab and save the fridge temperature before starting sweep
    # sweep.start_T = np.array([nf1.getTemp(), nf2.getTemp()])
    start_T = np.array([nf1.getResistance(), nf2.getResistance()])

    ## Set the VNA stimulus power and take a frequency sweep
    v.setPower(P_ctr)
    times, S21_real, S21_imag = v.timeDomain(f_res, npts, ifb=bdwt)

    ## Grab and save the fridge temperature after sweep
    # sweep.final_T = np.array([nf1.getTemp(), nf2.getTemp()])
    final_T = np.array([nf1.getResistance(), nf2.getResistance()])

    # ## Save the result to our class instance
    # sweep.frequencies = np.array(freqs)
    # sweep.S21realvals = np.array(S21_real)
    # sweep.S21imagvals = np.array(S21_imag)

    ## Write our class to a file (h5)
    # print("Storing data at:", sweep.save_hdf5(output_filename))

    ## Store the data in our file name (csv)
    v.storeData(freqs, S21_real, S21_imag, output_filename)

    ## Diagnostic text
    print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
    print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())
    print("Power scan complete.")
    return 0

if __name__ == "__main__":
    ## Initialize the NEXUS temperature servers
    nf1 = NEXUSTemps(server_ip="192.168.0.31",server_port=11031)
    nf2 = NEXUSTemps(server_ip="192.168.0.32",server_port=11032)

    ## Initialize the VNA
    v = VNA()

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Parameters of the power sweep (in dB)
    P_ctr  = args.power if args.power is not None else P_ctr
    f_res  = args.freq  if args.freq  is not None else f_res
    lapse  = args.time  if args.time  is not None else lapse
    srate  = args.rate  if args.rate  is not None else srate

    ## Recalculate inherited params
    bdwt   = srate * 1e6   ## IF Bandwith [sampling rate Hz]
    npts   = bdwt  * lapse ## N points to take

    ## Where to save the output data (hdf5 files)
    sweepPath = args.directory if args.directory is not None else sweepPath

    ## Create the output directories
    create_dirs()

    ## Run the time scan
    run_scan()