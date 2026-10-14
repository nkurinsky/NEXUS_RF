import sys
import numpy as np
import datetime
import argparse
from pathlib import Path
from copy import deepcopy

## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")
from VNAfunctions import *  #using the VNA to do a power sweep
from NEXUSFunctions import * #control NEXUS fridge

sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/BackendTools")
from VNAMeas import * #vna measurement class

_DEBUG_ = False

## Parameters of the power sweep (in dB)
## Set the VNA's frequency parameters
default_sweep_params = {
    "P_min": -55.0,       ## dBm
    "P_max":   0.0,       ## dBm
    "P_step":  5.0,       ## dBm
    "fcenter": np.array([3.9232, 4.0740, 4.8573, 6.1063, 6.3336, 6.7843])*1e9,  ## Hz
    "fspan":   2.0e6,     ## Hz
    "n_samps": 2e4 + 1,
    "n_avs":   5,
    "ifbw":    1e2,       ## Hz
}

## Where to save the output data (hdf5 files)
dataPath = Path('/data/PowerSweeps/VNA')  #VNA subfolder of TempSweeps

def parse_args():
    ## Instantiate the parser
    parser = argparse.ArgumentParser(description='Script to take a frequency sweep with the CMT VNA at a set of specified RF powers')

    ## Power scan optional arguments
    parser.add_argument('--P0', type=float, default=default_sweep_params["P_min"],
                        help='Minimum power for scan [dBm]')
    parser.add_argument('--P1', type=float, default=default_sweep_params["P_max"],
                        help='Maximum power for scan [dBm] (must be below 0 dBm)')
    parser.add_argument('--Ps', type=float, default=default_sweep_params["P_step"],
                        help='Power step for scan [dBm]')

    ## Frequency sweep optional arguments
    parser.add_argument('--Fc', type=float, nargs='+', default=None,
                        help='One or more center frequencies [Hz]')
    parser.add_argument('--Fs', type=float, default=default_sweep_params["fspan"],
                        help='Frequency span for scan [Hz]')
    parser.add_argument('--Fn', type=int, default=default_sweep_params["n_samps"],
                        help='Number of samples to take in frequency range')
    parser.add_argument('--Na', type=int, default=default_sweep_params["n_avs"],
                        help='Number of averages to take at each power')
    parser.add_argument('--Bw', type=int, default=default_sweep_params["ifbw"],
                        help='IF Bandwidth [Hz]')

    ## Data path optional arguments
    parser.add_argument('--d', type=str, default=dataPath,
                        help='Top-level directory for storing VNA data')

    ## Flag for debug mode
    parser.add_argument('--debug', action='store_true',
                        help='Print scan settings without taking measurements')

    ## Now read the arguments
    args = parser.parse_args()

    ## Reject NaN and infinity in all scalar numeric arguments
    for name in ("P0", "P1", "Ps", "Fs", "Fn", "Na", "Bw"):
        if not np.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")

    ## Require a valid ascending power range
    if args.P1 > 0:
        parser.error("--P1 must be <= 0 dBm")
    if args.P0 > args.P1:
        parser.error("--P0 must be <= --P1")
    if args.Ps <= 0:
        parser.error("--Ps must be > 0")

    ## Require valid measurement settings
    if args.Fs <= 0:
        parser.error("--Fs must be > 0 Hz")
    if args.Fn < 2:
        parser.error("--Fn must be >= 2")
    if args.Na < 1:
        parser.error("--Na must be >= 1")
    if args.Bw <= 0:
        parser.error("--Bw must be > 0 Hz")

    ## Normalize supplied or default center frequencies to a NumPy array
    args.Fc = np.array(
        default_sweep_params["fcenter"] if args.Fc is None else args.Fc,
        dtype=float,
    )

    ## Require finite center frequencies and positive lower sweep limits
    if not np.all(np.isfinite(args.Fc)):
        parser.error("--Fc values must be finite")
    if np.any(args.Fc <= args.Fs / 2.0):
        parser.error("Each --Fc must exceed --Fs / 2 so the sweep starts above 0 Hz")

    ## Validate the data path
    desired_path = Path(args.d)

    sweep_params = {
        "P_min":   args.P0,
        "P_max":   args.P1,
        "P_step":  args.Ps,
        "fcenter": np.array(args.Fc), ## must be a numpy array
        "fspan":   args.Fs,
        "f_min":   args.Fc - args.Fs/2.0,
        "f_max":   args.Fc + args.Fs/2.0,
        "n_samps": args.Fn,
        "n_avs":   args.Na,
        "ifbw":    args.Bw,
    }

    return sweep_params, desired_path, args.debug

def create_dirs(target_datapath):
    ## Sub directory definitions
    dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
    sweepPath = target_datapath / dateStr

    series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    seriesPath = sweepPath / series 

    target_datapath.mkdir(parents=True, exist_ok=True)
    sweepPath.mkdir(parents=True, exist_ok=True)
    seriesPath.mkdir(parents=True, exist_ok=True)

    print (f"Scan stored as series {series} in path {sweepPath}")
    return series, seriesPath

def run_scan(series, seriesPath, sweep_params):
    ## Diagnostic text
    #print("Current Fridge MC Temperature (mK): ", nf1.getTemp()*1e3)

    print("--Power Scan Settings-------")
    print("-   Start Power (dB):", sweep_params["P_min"])
    print("-     End Power (dB):", sweep_params["P_max"])
    print("-    Power Step (dB):", sweep_params["P_step"])
    print("- N Sweeps Avgeraged:", sweep_params["n_avs"])

    ## Get an array of all the powers to sweep at
    powers = np.arange(start = sweep_params["P_min"],
                       stop  = sweep_params["P_max"]+sweep_params["P_step"],
                       step  = sweep_params["P_step"])
    powers = powers[powers<=sweep_params["P_max"]]
    print("Scanning over powers (dB):", powers)

    ## Start the power loop
    for power in powers:
      print("Current Power (dBm):", str(round(power)))

      ## Create a filename for this sweep
      output_filename = seriesPath / f"Psweep_P{power}_{series}"

      ## Confirm we have valid keys
      if ("f_min" not in sweep_params) or ("f_max" not in sweep_params):
        sweep_params["f_min"] = sweep_params["fcenter"] - sweep_params["fspan"]/2.0
        sweep_params["f_max"] = sweep_params["fcenter"] + sweep_params["fspan"]/2.0

      ## Create a class to contain the sweep result
      sweep = VNAMeas(series.split("_")[0], series)
      sweep.power   = power
      sweep.n_avgs  = sweep_params["n_avs"]
      sweep.n_samps = sweep_params["n_samps"]
      sweep.f_min   = sweep_params["f_min"]
      sweep.f_max   = sweep_params["f_max"]
      sweep.ifbw    = sweep_params["ifbw"]

      ## Grab and save the fridge temperature before starting sweep
      sweep.start_T = np.array([ -1.0 ]) # nf1.getTemp() ])

      ## Set the VNA stimulus power and take a frequency sweep
      v.setPower(power)
      freqs, S21_real, S21_imag = v.takeSweep(sweep.f_min, sweep.f_max, 
                                              sweep.n_samps, 
                                              sweep.n_avgs, 
                                              ifb=sweep.ifbw)

      ## Grab and save the fridge temperature after sweep
      sweep.final_T = np.array([ -1.0 ]) # nf1.getTemp() ])

      ## Save the result to our class instance
      sweep.frequencies = np.array(freqs)
      sweep.S21realvals = np.array(S21_real)
      sweep.S21imagvals = np.array(S21_imag)

      ## Write our class to a file (h5)
      print("Storing data at:", sweep.save_hdf5(str(output_filename)))

      ## Store the data in our file name (csv)
      #v.storeData(freqs, S21_real, S21_imag, output_filename)

    ## Diagnostic text
    # print("Current Fridge MC Temperature (mK): ", nf1.getTemp()*1e3)
    print("Power scan complete.")
    return 0

if __name__ == "__main__":
    #start = time.time()
    ## Initialize the NEXUS temperature servers
    #nf1 = NEXUSHeater(server_ip="192.168.0.34",server_port=11034)

    ## Parse command line arguments to set parameters
    sweep_params, dataPath, _DEBUG_ = parse_args()

    ## Initialize the VNA
    if not _DEBUG_: v = VNA()

    ## Loop over every tone
    for i in np.arange(len(sweep_params["fcenter"])):

        ## Copy the sweep params and update the frequency limits
        this_scan_params = deepcopy(sweep_params)
        this_scan_params["fcenter"] = sweep_params["fcenter"][i]
        this_scan_params["f_min"]   = sweep_params["f_min"][i]
        this_scan_params["f_max"]   = sweep_params["f_max"][i]

        if _DEBUG_:
            print(f"Center f [GHz]: {this_scan_params['fcenter']/1e9}")
            print(f"Minimum f [GHz]: {this_scan_params['f_min']/1e9}")
            print(f"Maximum f [GHz]: {this_scan_params['f_max']/1e9}")

        else:
            ## Create the output directories
            series, seriesPath = create_dirs(dataPath)
        
            ## Run the power scan
            run_scan(series, seriesPath, sweep_params=this_scan_params)

    #end=time.time()
    #print(-(start-end)/60.,"minutes")
