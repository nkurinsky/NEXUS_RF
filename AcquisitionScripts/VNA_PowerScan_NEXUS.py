import sys, os
#from time import sleep, time
#import time
import numpy as np
import datetime
import argparse

## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")
from VNAfunctions import *  #using the VNA to do a power sweep
from NEXUSFunctions import * #control NEXUS fridge

sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/BackendTools")
from VNAMeas import * #vna measurement class

## Parameters of the power sweep (in dB)
P_min  = -55.0
P_max  = -20.0
P_step =   1.0

## Set the VNA's frequency parameters
freqmin = 4.24205e9 # 4.24212e9   ## Hz
freqmax = 4.24225e9 # 4.24262e9   ## Hz
n_samps = 5e4

## How many readings to take at each step of the sweep
n_avs = 10

## Where to save the output data (hdf5 files)
dataPath = '/data/PowerSweeps/VNA'  #VNA subfolder of TempSweeps

## Sub directory definitions
dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath = dataPath + '/' + dateStr

series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = sweepPath + '/' + series 

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Script to take a frequency sweep with the CMT VNA at a set of specified RF powers')

    # Power scan optional arguments
    parser.add_argument('--P0', type=float,
                        help='Minimum power for scan [dBm]')
    parser.add_argument('--P1', type=float,
                        help='Maximum power for scan [dBm] (must be below -10)')
    parser.add_argument('--Ps', type=float,
                        help='Power step for scan [dBm]')

    # Frequency sweep optional arguments
    parser.add_argument('--F0', type=float,
                        help='Minimum frequency for sweep [Hz]')
    parser.add_argument('--F1', type=float,
                        help='Maximum frequency for scan [Hz] (must be above minimum frequency)')
    parser.add_argument('--Fn', type=int,
                        help='Number of samples to take in frequency range')
    parser.add_argument('--Na', type=int,
                        help='Number of averages to take at each power')

    # Data path optional arguments
    parser.add_argument('--d', type=str,
                        help='Top-level directory for storing VNA data')

    # Now read the arguments
    args = parser.parse_args()

    # Do some conditional checks
    if (args.P1 is not None):
        if (args.P1 > -10):
            print(args.P1, "dBm is too large, setting P_max to -10")
            args.P1 = -10.0

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

def run_scan():

    ## Diagnostic text
    print("Current Fridge MC Temperature (mK): ", nf1.getTemp()*1e3)

    print("--Power Scan Settings-------")
    print("-   Start Power (dB):", P_min)
    print("-     End Power (dB):", P_max)
    print("-    Power Step (dB):", P_step)
    print("- N Sweeps Avgeraged:", n_avs)

    ## Get an array of all the powers to sweep at
    powers = np.arange(start = P_min,
                       stop  = P_max+P_step,
                       step  = P_step)
    print("Scanning over powers (dB):", powers)

    ## Start the power loop
    for power in powers:
      print("Current Power (dBm):", str(round(power)))

      ## Create a filename for this sweep
      output_filename = seriesPath +"/Psweep_P"+str(power)+"_"+series

      ## Create a class to contain the sweep result
      sweep = VNAMeas(dateStr, series)
      sweep.power   = power
      sweep.n_avgs  = n_avs
      sweep.n_samps = n_samps
      sweep.f_min   = freqmin
      sweep.f_max   = freqmax

      ## Grab and save the fridge temperature before starting sweep
      sweep.start_T = np.array([ nf1.getTemp() ])

      ## Set the VNA stimulus power and take a frequency sweep
      v.setPower(power)
      freqs, S21_real, S21_imag = v.takeSweep(freqmin, freqmax, n_samps, n_avs)

      ## Grab and save the fridge temperature after sweep
      sweep.final_T = np.array([ nf1.getTemp() ])

      ## Save the result to our class instance
      sweep.frequencies = np.array(freqs)
      sweep.S21realvals = np.array(S21_real)
      sweep.S21imagvals = np.array(S21_imag)

      ## Write our class to a file (h5)
      print("Storing data at:", sweep.save_hdf5(output_filename))

      ## Store the data in our file name (csv)
      #v.storeData(freqs, S21_real, S21_imag, output_filename)

    ## Diagnostic text
    print("Current Fridge MC Temperature (mK): ", nf1.getTemp()*1e3)
    print("Power scan complete.")
    return 0

if __name__ == "__main__":
    #start = time.time()
    ## Initialize the NEXUS temperature servers
    nf1 = NEXUSHeater(server_ip="192.168.0.34",server_port=11034)

    ## Initialize the VNA
    v = VNA()

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Parameters of the power sweep (in dB)
    P_min  = args.P0 if args.P0 is not None else P_min
    P_max  = args.P1 if args.P1 is not None else P_max
    P_step = args.Ps if args.Ps is not None else P_step

    ## Set the VNA's frequency parameters
    freqmin = args.F0 if args.F0 is not None else freqmin
    freqmax = args.F1 if args.F1 is not None else freqmax
    n_samps = args.Fn if args.Fn is not None else n_samps

    ## How many readings to take at each step of the sweep
    n_avs   = args.Na if args.Na is not None else n_avs

    ## Where to save the output data (hdf5 files)
    dataPath = args.d if args.d is not None else dataPath

    ## Create the output directories
    create_dirs()

    ## Run the power scan
    run_scan()

    #end=time.time()
    #print(-(start-end)/60.,"minutes")
