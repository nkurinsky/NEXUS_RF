import sys, os
from time import sleep
import numpy as np
import datetime

# ## Point to the backend function scripts
# sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")

# from VNAfunctions import *  #using the VNA to do a power sweep
# from NEXUSFunctions import * #control NEXUS fridge
# from VNAMeas import * #vna measurement class

## Parameters of the power sweep (in dB)
P_min  = -45.0
P_max  = -20.0
P_step =   5.0

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

def parse_arguments():
    ## Count the number of arguments provided to python interpreter
    ## First argument is always script name
    arg_len = len(sys.argv) - 1
    arg_arr = sys.argv[1:]

    ## If none, do nothing else
    if arg_len==0:
        return 0

    ## Otherwise, read them
    for i in np.arange(arg_len-1):

        if not arg_arr[i][0]=="-":
            continue

        if arg_arr[i] == "-P0":
            try:
                print("Setting P_min to",arg_arr[i+1],"dBm")
                P_min = float(arg_arr[i+1])
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for P_min")
            continue

        if arg_arr[i] == "-P1":
            try:
                print("Setting P_max to",arg_arr[i+1],"dBm")
                P_max = float(arg_arr[i+1])
                if (P_max > -10):
                    print(P_max, "dBm is too large, setting P_max to -10")
                    P_max = -10.0
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for P_max, using default.")
            continue

        if arg_arr[i] == "-Ps":
            try:
                print("Setting P_step to",arg_arr[i+1],"dBm")
                P_step = float(arg_arr[i+1])
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for P_step, using default.")
            continue

        if arg_arr[i] == "-F0":
            try:
                print("Setting freqmin to",arg_arr[i+1],"Hz")
                freqmin = float(arg_arr[i+1])
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for freqmin, using default.")
            continue

        if arg_arr[i] == "-F1":
            try:
                print("Setting freqmax to",arg_arr[i+1],"Hz")
                freqmax = float(arg_arr[i+1])
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for freqmax, using default.")
            continue

        if arg_arr[i] == "-Ns":
            try:
                print("Setting n_samps to",arg_arr[i+1])
                n_samps = int(arg_arr[i+1])
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for n_samps, using default.")
            continue

        if arg_arr[i] == "-Na":
            try:
                print("Setting n_avs to",arg_arr[i+1])
                n_avs = int(arg_arr[i+1])
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for n_avs, using default.")
            continue

        if arg_arr[i] == "-d":
            try:
                print("Setting dataPath to",arg_arr[i+1])
                dataPath = arg_arr[i+1]
            except:
                print("\"",arg_arr[i+1],"\" is not a valid value for n_avs, using default.")
            continue

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
    print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
    print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())

    print("--Power Scan Settings-------")
    print("-   Start Power (dB):", P_min)
    print("-     End Power (dB):", P_max)
    print("-    Power Step (dB):", P_step)
    print("- N Points Avgeraged:", n_avs)

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
      sweep.start_T = np.array([nf1.getTemp(), nf2.getTemp()])

      ## Set the VNA stimulus power and take a frequency sweep
      v.setPower(power)
      freqs, S21_real, S21_imag = v.takeSweep(freqmin, freqmax, n_samps, n_avs)

      ## Grab and save the fridge temperature after sweep
      sweep.final_T = np.array([nf1.getTemp(), nf2.getTemp()])

      ## Save the result to our class instance
      sweep.frequencies = np.array(freqs)
      sweep.S21realvals = np.array(S21_real)
      sweep.S21imagvals = np.array(S21_imag)

      ## Write our class to a file (h5)
      print("Storing data at:", sweep.save_hdf5(output_filename))

      ## Store the data in our file name (csv)
      v.storeData(freqs, S21_real, S21_imag, output_filename)

    ## Diagnostic text
    print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
    print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())
    print("Power scan complete.")
    return 0

if __name__ == "__main__":
    # # Initialize the NEXUS temperature servers
    # nf1 = NEXUSTemps(server_ip="192.168.0.31",server_port=11031)
    # nf2 = NEXUSTemps(server_ip="192.168.0.32",server_port=11032)

    # ## Initialize the VNA
    # v = VNA()

    ## Parse command line arguments to set parameters
    parse_arguments()

    # ## Create the output directories
    # create_dirs()

    # ## Run the power scan
    # run_scan()