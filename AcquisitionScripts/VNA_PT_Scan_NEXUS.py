import sys, os
import numpy as np
import datetime
import argparse

from time import sleep

## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")

from VNAfunctions import *  #using the VNA to do a power sweep
from NEXUSFunctions import * #control NEXUS fridge
from VNAMeas import * #vna measurement class

## Parameters of the power sweep (in dB)
P_min  = -55
P_max  = -15
P_step =   5

## Set the VNA's frequency parameters
freqmin = 4.242005e9 # 4.244585e9   ## Hz
freqmax = 4.242355e9 # 4.244936e9   ## Hz
n_samps = 14e3

## How many readings to take at each step of the sweep
n_avs = 3

## Temperature scan settings [K]
Temp_base =  11e-3
Temp_min  =  20e-3
Temp_max  = 350e-3
Temp_step =  10e-3

## Temperature stabilization params
tempTolerance =   1e-4     ## K
sleepTime     =  30        ## sec
stabletTime   = 180        ## sec

## Create the temperature array
Temps = np.arange(Temp_min,Temp_max+Temp_step,Temp_step)

## Use this if starting at the top temperature
Temps = Temps[::-1] 
if (Temp_base) < Temps[-1]:
    Temps = np.append(Temps,Temp_base)

# ## Use this if starting at base temperature
# if (Temp_base) < Temps[0]:
#     Temps = np.append(Temp_base,Temps)

## Where to save the output data (hdf5 files)
dataPath = '/data/Tempsweeps/VNA'  #VNA subfolder of Tempsweeps

## Sub directory definitions
dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath = os.path.join(dataPath , dateStr)

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

    return 0

def create_series_dir():

    series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    seriesPath = os.path.join(sweepPath , series) 

    if not os.path.exists(seriesPath):
        os.makedirs(seriesPath)
    print ("Scan stored as series "+series+" in path "+sweepPath)

    return series, seriesPath

def temp_change_and_wait(new_sp_K,nf_inst):

    print("CHANGING SETPOINT TO",new_sp*1e3,"mK")
    nf_inst.setSP(new_sp)

    cTemp=float(nf_inst.getTemp())
    print("Waiting for Fridge to Reach Temperature")
    print("Monitoring temp every",sleepTime,"seconds")
    print("...",cTemp*1e3,"mK")
    terr = new_sp_K-cTemp
    while(np.abs(terr) > tempTolerance):
        sleep(sleepTime)
        try:
            cTemp=float(nf_inst.getTemp())
            terr = new_sp_K-cTemp
            print("...",cTemp*1e3,"("+str(terr*1e3)+") mK")
        except:
            print("Socket Failed, skipping reading")

    print("Holding at current temp for",stabletTime,"seconds")
    sleep(stabletTime)

    print("Done.")
    return 0

def run_power_scan(currTemp, seriesPath, nf_inst):

    ## Diagnostic text
    #print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
    #print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())

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
      output_filename = os.path.join(seriesPath,"TPsweep"+"_T"+str(round(cTemp,5)*1e3)+"_P"+str(power)+"_"+series)

      ## Create a class to contain the sweep result
      sweep = VNAMeas(dateStr, series)
      sweep.power   = power
      sweep.n_avgs  = n_avs
      sweep.n_samps = n_samps
      sweep.f_min   = freqmin
      sweep.f_max   = freqmax

      ## Grab and save the fridge temperature before starting sweep
      # sweep.start_T = np.array([nf1.getTemp(), nf2.getTemp()])
      sweep.start_T = np.array([float(nf_inst.getTemp()), -1.0]) # , nf2.getResistance()])

      ## Set the VNA stimulus power and take a frequency sweep
      v.setPower(power)
      freqs, S21_real, S21_imag = v.takeSweep(freqmin, freqmax, n_samps, n_avs)

      ## Grab and save the fridge temperature after sweep
      # sweep.final_T = np.array([nf1.getTemp(), nf2.getTemp()])
      sweep.final_T = np.array([float(nf_inst.getTemp()), -1.0]) # np.array([-1.0,-1.0]) #[nf1.getResistance(), nf2.getResistance()])

      ## Save the result to our class instance
      sweep.frequencies = np.array(freqs)
      sweep.S21realvals = np.array(S21_real)
      sweep.S21imagvals = np.array(S21_imag)

      ## Write our class to a file (h5)
      print("Storing data at:", sweep.save_hdf5(output_filename))

      ## Store the data in our file name (csv)
      #v.storeData(freqs, S21_real, S21_imag, output_filename)

    ## Diagnostic text
    #print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
    #print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())
    print("Power scan complete.")
    return 0

if __name__ == "__main__":
    ## Initialize the NEXUS temperature servers
    nf3 = NEXUSTemps(server_ip="192.168.0.34",server_port=11034)

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

    ## Print some diagnostic text
    SP = float(nf3.getSP())
    print("Starting Set Point:",SP)
    print("Scan Settings")
    print("         Start Temp:",Temps[ 0]*1e3,"mK")
    print("           End Temp:",Temps[-1]*1e3,"mK")
    print("          Temp Step:",Temp_step*1e3,"mK")
    print("     Temp Tolerance:",tempTolerance*1e3,"mK")
    print("          Hold Time:",holTemp_stepime,"s")
    print("   Reading Interval:",sleepTime,"s")

    ## Run the temperature scan
    for T in Temps:
        
        ## Change the fridge temperature
        temp_change_and_wait(T, nf3)

        ## Create a new directory
        series, seriesPath = create_series_dir()

        ## Run a power scan
        run_power_scan(T, seriesPath, nf3)

    ## Go back to base temperature
    print("Reverting to base temperature")
    nf.setSP(Temp_base)


