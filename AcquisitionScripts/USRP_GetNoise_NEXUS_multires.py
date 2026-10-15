## Import the relevant modules
import sys, os
import time, datetime
import argparse
import h5py, json
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
    "LO"      : [3.95e9, 4.05e9, 4.85e9], #, 6.15e9, 6.35e9, 6.80e9],         ## [Hz] Round numbers, no finer than 50 MHz. Always must be list, same length as res freqs
}

## Set VNA sweep parameters
default_vna_params = {
    "f_span_kHz" : 2000,        ## Symmetric about the center frequency
    "points"     : 20000,       ## Defined such that we look at 100 Hz windows
    "duration"   : 150,         ## [Sec] ## IF_BW ~= points / duration
}

## Set timestream parameters
default_readout_params = {
    "res"            : [3.9232, 4.0740, 4.8573], #, 6.1063, 6.3336, 6.7843],          ## Always must be list [GHz]
    "tracking_offset_MHz" : 10.0,           ## Symmetric cleaning-tone offset from resonance [MHz]
    "powers"         : np.arange(start=-60,stop=-10,step=5),  ## dBm
    "duration"       : 30,                  ## [Sec] ## IF_BW ~= points / duration
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

    ## Allow argv to be supplied directly for testing
    parser = argparse.ArgumentParser(
        description="Acquire USRP noise timestreams using the GPU_SDR backend."
    )

    ## DAQ settings; LO accepts one or more values in MHz
    parser.add_argument("--rate", "-r", type=float,
                        default=default_daq_params["rate"] / 1e6)
    parser.add_argument("--txgain", "-tx", type=float,
                        default=default_daq_params["tx_gain"])
    parser.add_argument("--rxgain", "-rx", type=float,
                        default=default_daq_params["rx_gain"])
    parser.add_argument(
        "--LOfrq", "--LO", "-f", dest="lo_mhz",
        type=float, nargs="+",
        default=np.atleast_1d(default_daq_params["LO"]) / 1e6,
        help="LO frequencies [MHz]: one shared value or one per resonator",
    )

    ## VNA settings
    parser.add_argument("--VNAfspan", "-fv", type=float,
                        default=default_vna_params["f_span_kHz"])
    parser.add_argument("--points", "-p", type=int,
                        default=default_vna_params["points"])
    parser.add_argument("--timeVNA", "-Tv", type=float,
                        default=default_vna_params["duration"])

    ## Readout frequencies and powers accept one or more values
    parser.add_argument(
        "--resFreq", "--res", "-fr", dest="res_ghz",
        type=float, nargs="+",
        default=default_readout_params["res"],
        help="Resonant frequencies [GHz]",
    )
    ## Shared symmetric offset; explicit pairs override this value
    parser.add_argument(
        "--trackOffset", type=float,
        default=default_readout_params["tracking_offset_MHz"],
        help="Shared tracking offset [MHz]; creates tones at res +/- offset",
    )
    ## Repeat once per resonator for signed lower/upper offsets
    parser.add_argument(
        "--trackPair", type=float, nargs=2, action="append", default=None,
        metavar=("LOW_MHZ", "HIGH_MHZ"),
        help="Signed tracking offsets [MHz]; repeat once per LO/res pair",
    )
    parser.add_argument(
        "--power", "-P", type=float, nargs="+",
        default=default_readout_params["powers"],
        help="RF powers [dBm]",
    )
    parser.add_argument("--timeNoise", "-Tn", type=float,
                        default=default_readout_params["duration"])

    ## Omitting this option uses defaults; supplying it without values gives []
    parser.add_argument(
        "--calDeltas", "-cd", type=float, nargs="*",
        default=default_readout_params["cal_deltas"],
        help="Calibration offsets; supply without values for an empty list",
    )
    parser.add_argument("--calTime", "-Tc", type=float,
                        default=default_readout_params["cal_lapse_sec"])

    ## Data path optional arguments
    parser.add_argument('--d', type=str, 
        default = dataPath,
        help = f'Top-level directory for storing VNA data (default: {dataPath})')

    # ## Flag for debug mode
    # parser.add_argument('--debug', action='store_true',
    #     help='Print scan settings without taking measurements')
    
    ## Now read the arguments
    args = parser.parse_args()

    ## Do some conditional checks

    ## Reject nonfinite scalar values
    scalar_names = (
        "rate", "txgain", "rxgain", "VNAfspan",
        "points", "timeVNA", "timeNoise", "calTime",
    )
    for name in scalar_names:
        if not np.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite")

    ## Normalize arrays and validate both CLI values and defaults
    def numeric_array(values, option, allow_empty=False):
        result = np.array(values, dtype=float, ndmin=1, copy=True)
        if result.ndim != 1:
            parser.error(f"{option} must be a one-dimensional list")
        if not allow_empty and result.size == 0:
            parser.error(f"{option} requires at least one value")
        if not np.all(np.isfinite(result)):
            parser.error(f"{option} values must be finite")
        return result

    lo_mhz = numeric_array(args.lo_mhz, "--LOfrq")
    res_ghz = numeric_array(args.res_ghz, "--resFreq")
    powers = numeric_array(args.power, "--power")
    cal_deltas = numeric_array(
        args.calDeltas, "--calDeltas", allow_empty=True
    )

    ## Always include the central measurement, including for an empty array
    if not np.any(cal_deltas == 0.0):
        cal_deltas = np.append(cal_deltas, 0.0)

    ## Acquire offsets in ascending order and remove duplicate measurements
    cal_deltas = np.unique(cal_deltas)

    ## An empty list skips offset calibration but still acquires the central tone
    if cal_deltas.size == 0:
        cal_deltas = np.array([0.0], dtype=float)

    ## Require one shared LO or exactly one LO per resonator
    if lo_mhz.size == 1:
        lo_mhz = np.full(res_ghz.size, lo_mhz[0])
    elif lo_mhz.size != res_ghz.size:
        parser.error(
            "--LOfrq requires one value or the same number as --resFreq"
        )

    ## Validate the shared tracking offset
    if not np.isfinite(args.trackOffset) or args.trackOffset <= 0:
        parser.error("--trackOffset must be finite and > 0 MHz")

    ## Build one signed lower/upper pair per resonator
    if args.trackPair is None:
        offsets_mhz = np.tile(
            [-args.trackOffset, args.trackOffset], (res_ghz.size, 1)
        )
    else:
        offsets_mhz = np.array(args.trackPair, dtype=float)
        if offsets_mhz.shape != (res_ghz.size, 2):
            parser.error("--trackPair must be supplied once per resonator")
        if not np.all(np.isfinite(offsets_mhz)):
            parser.error("--trackPair offsets must be finite")
        if np.any(offsets_mhz[:, 0] >= 0) or np.any(offsets_mhz[:, 1] <= 0):
            parser.error("--trackPair requires a negative then positive offset")

    ## Convert offsets to absolute tracking frequencies for each resonator [Hz]
    with np.errstate(over="ignore", invalid="ignore"):
        tracking = res_ghz[:, None] * 1e9 + offsets_mhz * 1e6
    if not np.all(np.isfinite(tracking)) or np.any(tracking <= 0):
        parser.error("Tracking frequencies must be finite and > 0 Hz")

    ## Require each LO to be a multiple of 50 MHz
    nearest_grid_mhz = np.round(lo_mhz / 50.0) * 50.0
    if not np.all(np.isclose(
        lo_mhz, nearest_grid_mhz, rtol=0.0, atol=1e-6
    )):
        parser.error("--LOfrq values must be multiples of 50 MHz")

    ## Normalize insignificant floating-point differences to the exact grid
    lo_mhz = nearest_grid_mhz

    ## Require each LO to be within 150 MHz of its paired resonator
    res_mhz = res_ghz * 1e3
    if np.any(np.abs(lo_mhz - res_mhz) > 150.0 + 1e-6):
        parser.error(
            "Each --LOfrq must be within 150 MHz of its paired --resFreq"
        )

    ## Preserve the existing script's sampling-rate and span limits
    if not 0 < args.rate <= 100:
        parser.error("--rate must be > 0 and <= 100 Msps")
    if not 0 < args.VNAfspan <= 10_000:
        parser.error("--VNAfspan must be > 0 and <= 10000 kHz")
    if args.points < 2:
        parser.error("--points must be >= 2")

    ## Require positive measurement durations
    for name in ("timeVNA", "timeNoise", "calTime"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be > 0 seconds")

    ## Validate frequency values before unit conversion
    if np.any((lo_mhz <= 0) | (lo_mhz > 6000)):
        parser.error("--LOfrq values must be > 0 and <= 6000 MHz")
    if np.any(res_ghz <= 0):
        parser.error("--resFreq values must be > 0 GHz")

    ## Preserve the existing script's requested power limits
    if np.any((powers < -70) | (powers > -15)):
        parser.error("--power values must be between -70 and -15 dBm")

    ## Convert units once and reject overflow in derived frequencies
    with np.errstate(over="ignore", invalid="ignore"):
        lo_hz = lo_mhz * 1e6
        res_hz = res_ghz * 1e9
        span_hz = args.VNAfspan * 1e3
        lower = res_hz - span_hz / 2
        upper = res_hz + span_hz / 2

    if not np.all(np.isfinite(lower) & np.isfinite(upper)):
        parser.error("Derived sweep frequencies must be finite")
    if np.any(lower <= 0):
        parser.error("Every VNA sweep must start above 0 Hz")

    ## Validate the data path
    desired_path = Path(args.d)

    ## Retain the existing dictionary units and key names
    params = {
        "daq": {
            "rate": args.rate * 1e6,
            "tx_gain": args.txgain,
            "rx_gain": args.rxgain,
            "LO": lo_hz,
        },
        "vna": {
            "f_span_kHz": args.VNAfspan,
            "points": args.points,
            "duration": args.timeVNA,
        },
        "ro": {
            "res": res_ghz,
            "tracking_tones": tracking,
            "powers": powers,
            "duration": args.timeNoise,
            "cal_deltas": cal_deltas,
            "cal_lapse_sec": args.calTime,
        },
    }

    return params, desired_path #, args.debug

if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    params, desired_path = parse_args()
    desired_path = desired_path.resolve()

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    try:
        ## Create a dictionary to track metadata about results
        scan_info = {}
        first_series = "None"

        ## Now look for all LO/res pairs and iterate over them
        for j, (lo_hz, res_ghz) in enumerate(zip(params["daq"]["LO"], params["ro"]["res"])):

            scan_info[f"R{j}_F{res_ghz}"] = {
                "LO_MHz": lo_hz/1e6,
                "series": [],
                "powers": [],
            }

            ## Get the run parameter dictionary
            daq_params = daq.generate_daq_params(
                front_end="A", 
                f_center_Hz=res_ghz*1e9, 
                rate=params["daq"]["rate"], 
                tx_gain=params["daq"]["tx_gain"], 
                rx_gain=params["daq"]["rx_gain"], 
                LO_freq=lo_hz, 
                vna_duration=params["vna"]["duration"], 
                vna_iterations=1, 
                f_span_Hz=params["vna"]["f_span_kHz"]*1e3,
                vna_npoints=params["vna"]["points"], 
                stream_duration=params["ro"]["duration"], 
                calibration_duration=params["ro"]["cal_lapse_sec"],
                cal_deltas=params["ro"]["cal_deltas"],
                tracking_tones=params["ro"]["tracking_tones"][j],
                delay_duration=10.0, 
            )

            ## Loop over the powers considered
            powers = params["ro"]["powers"]
            for i in np.arange(len(powers)):

                ## Define and create the output directories
                date_path, series_path, series = daq.get_paths(desired_path)
                os.chdir(series_path) ## When doing this, no need to provide subfolder
                if (j==0) and (i==0): first_series = series

                ## Ensure the power doesn't go above -25 dBm
                ## Due to power splitting across tones

                ## Preserve the user's gain and compensate in synthesized power
                this_power = float(powers[i])
                tx_gain = params["daq"]["tx_gain"]
                rf_power = this_power - tx_gain

                ## Reject combinations that exceed the synthesized power limit
                if rf_power > -25.0:
                    rf_power = -25.0
                    tx_gain = this_power - rf_power

                daq_params["rf_power"] = rf_power
                daq_params["tx_gain"] = tx_gain

                ## Instantiate an output file
                with h5py.File('noise_averages_'+series+'.h5','w') as fyle:
                    daq.run_full_suite(series, daq_params, [res_ghz], run_type="Noise", h5_group_obj=fyle, subrun_id=i)

                ## Update the metadata tracker
                scan_info[f"R{j}_F{res_ghz}"]["series"].append(series)
                scan_info[f"R{j}_F{res_ghz}"]["powers"].append(this_power)

        ## Dump the metadata to a json file
        json_fname = f"usrp-noise-metadata-{first_series}.json"
        with open(str(desired_path / json_fname), "w") as file:
            json.dump(scan_info, file, indent=4)

    ## Disconnect from the USRP server
    finally:
        ## Disconnect even after an acquisition or metadata-writing failure
        u.Disconnect()
