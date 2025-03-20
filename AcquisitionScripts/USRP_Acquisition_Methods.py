import sys, os
import time, datetime

import numpy as np

from KIPD_Analysis import vna_file_fit, read_timestream, avg_timestream

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../DeviceControl/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")
        exit()

def get_paths(base_path, dt=None, create_dirs=True):
    ## Check that a valid datetime has been passed
    if (dt is None) or (type(dt) != type(datetime.datetime.now())):
        dt = datetime.datetime.now()

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    ## Get the first-level directory (for the date)
    date_str  = str(dt.strftime('%Y%m%d')) #sweep date
    date_path = os.path.join(base_path,date_str)
    if not os.path.exists(date_path):
        os.makedirs(date_path)

    ## Get the second-level directory (for the series)
    series      = str(dt.strftime('%Y%m%d_%H%M%S'))
    series_path = os.path.join(date_path,series)
    if not os.path.exists(series_path):
        os.makedirs(series_path)

    print ("Scan to be stored as series "+series+" in path "+date_path)

    return date_path, series_path, series

def generate_daq_params(front_end="A", rate=100e6, tx_gain=0.0, rx_gain=17.0, LO_freq=4.25e9, rf_power=-30.0, 
                        delay_duration=10.0, vna_duration=15.0, stream_duration=30.0, calibration_duration=5.0,
                        f_center_Hz=4.241265e9, f_span_Hz=200.0e3, f_start_Hz=None, f_stop_Hz=None, 
                        vna_npoints=2000, vna_iterations=1, cal_deltas=np.linspace(start=-0.05, stop=0.05, num=3),
                        tracking_tones=np.array([4.231e9,4.251e9])):
    params = {
        "front_end" : front_end,
        "rate"      : rate,
        "tx_gain"   : tx_gain,
        "rx_gain"   : rx_gain,
        "LO_freq"   : LO_freq,
        "rf_power"  : rf_power,

        "delay"     : {
            "duration_s" : delay_duration,
        },

        "vna"       : {
            "duration_s"     : vna_duration,
            "f_center_Hz"    : f_center_Hz,
            "f_span_Hz"      : f_span_Hz,
            "f_start_Hz"     : f_start_Hz,
            "f_stop_Hz"      : f_stop_Hz,
            "n_points"       : vna_npoints,
            "iterations"     : vna_iterations,
        },

        "stream"    : {
            "duration_s"     : stream_duration,
            "calib_s"        : calibration_duration,
            "cal_deltas"     : cal_deltas,
            "track_tones_Hz" : tracking_tones,
        },
    }

    return params



def run_delay(series, run_params, h5_group_obj=None, delay_over_s=None):
    ''' Runs a line delay measurement for the LO and returns the delay result in seconds.
        If a delay is specified by the user, that is loaded and used in subsequent measurements. 
        No line delay measurement is made, and no line delay is written to the data h5 file.
        If no delay is specified by the user, the code checks to see if a line delay already
        has been loaded into the USRP. If it does, it loads it. If it does not, a line delay measurement is made.
    Necessary input parameters
     - run_params["front_end"]      :   String identifier for front end of USRP used for acquisiton
     - run_params["rate"]           :   Digitization rate in samples per second
     - run_params["tx_gain"]        :   Tx gain applied internal to USRP
     - run_params["rx_gain"]        :   Rx gain applied internal to USRP
     - run_params["freq"]           :   Stimulus frequency
     - run_params["delay"]["duration_s"]    :   Duration over which to acquire the line delay measurement
    Optional input parameters
     - delay_over_s                 :   Forced value of line delay, in seconds
     - h5_group_obj                 :   H5 data group (in summary file) in which to save the metadata (delay value)
    '''

    ## Check if the user provided a line delay 
    if delay_over_s is not None: # in run_params.keys():
        print("Using user-specified line delay:", delay_over_s, "s")
        u.set_line_delay(run_params["rate"], delay_over_s*1e9) ## passes value in ns
        delay = delay_over_s ## delay still saved as seconds
        filename = None

    ## Now, determine if we need to run a line delay measurement, or if it's already been done
    else:
        search_key = str(int(run_params["rate"]/1e6))
        
        ## If this digitization rate already has a line delay, load it
        if search_key in u.LINE_DELAY.keys():
            print("Line delay found in current USRP/GPU_SDR configuration.")
            delay = u.LINE_DELAY[search_key]*1e-9 ## line delay is stored as ns, pull it out as sec

        ## If not, make a measurement
        else:
            print("Line delay not found in current USRP/GPU_SDR configuration. Measuring line delay.")

            ## Generate the file name for the delay data, and run it
            outfname = "USRP_Delay_"+series

            filename = u.measure_line_delay(run_params["rate"], run_params["LO_freq"], run_params["front_end"], USRP_num=0, 
                tx_gain=run_params["tx_gain"], 
                rx_gain=run_params["rx_gain"], 
                compensate = True, 
                duration = run_params["delay"]["duration_s"],
                output_filename=outfname, 
                subfolder=None)
            print("Done.")

            print("Analyzing line delay file...")
            delay = u.analyze_line_delay(filename, False)
            print("Done.")

            print("Writing line delay to file...")
            u.write_delay_to_file(filename, delay)
            print("Done.")

            print("Loading line delay from file...")
            u.load_delay_from_file(filename)
            print("Done.")

    ## Store the line delay as metadata in our noise file
    if h5_group_obj is not None:
        h5_group_obj.attrs.create("delay_ns", delay * 1e9)

    ## Add the delay to the params dict for later use
    run_params["delay"]["value_s"] = delay

    return delay, filename


def run_vna(series, run_params, res_search_freqs_GHz=None, h5_group_obj=None, cooltime_s=5):


    ## Determine how many equivalent tones to split the DAC power output into to achieve the requested output power
    N_power = np.power(10.,(((-1*run_params["rf_power"])-14)/20.))
    pwr_clc = np.round(-14-20*np.log10(N_power),2)
    print("To achieve",pwr_clc,"dBm of power for VNA scan, must split DAC power into",N_power,"equivalent tones")

    ## Check for validity of the equivalent number of tones to use for power compensation
    if (N_power == 1): N_power = None
    print("Using", N_power, "tones for Multitone_compensation")

    ## Do some math to find the frequency span for the VNA relative to the LO frequency
    ## If we tell it start and stop frequencies, override the center and span
    if (run_params["vna"]["f_start_Hz"] is not None) and (run_params["vna"]["f_stop_Hz"] is not None):
        print("User specified VNA start and stop frequencies: (",
            run_params["vna"]["f_start_Hz"]/1e6, "," ,
            run_params["vna"]["f_stop_Hz"]/1e6,", MHz). Overriding center and span frequencies")

        run_params["vna"]["f_center_Hz"] = np.mean([run_params["vna"]["f_start_Hz"],run_params["vna"]["f_stop_Hz"]])
        run_params["vna"]["f_span_Hz"]   = run_params["vna"]["f_stop_Hz"] - run_params["vna"]["f_start_Hz"]

    ## Now re-calculate the start and stop frequencies relative to the LO, based on center and span
    print("F span (VNA):",run_params["vna"]["f_span_Hz"],"Hz")
    fVNAmin = run_params["vna"]["f_center_Hz"] - (run_params["vna"]["f_span_Hz"]/2.)
    fVNAmax = run_params["vna"]["f_center_Hz"] + (run_params["vna"]["f_span_Hz"]/2.)
    print("VNA spans", fVNAmin/1e6, "MHz to", fVNAmax/1e6, "MHz")
    f0 = fVNAmin - run_params["LO_freq"]
    f1 = fVNAmax - run_params["LO_freq"]
    print("Relative to LO: start", f0, "Hz; stop",f1,"Hz")

    ## Create the boilerplate file name and run the VNA scan
    outfname = "USRP_VNA_"+series

    print("Starting single VNA run...")
    vna_filename  = u.Single_VNA(start_f = f0, last_f = f1, 
        measure_t = run_params["vna"]["duration_s"],
        n_points  = run_params["vna"]["n_points"], 
        tx_gain   = run_params["tx_gain"], 
        rx_gain   = run_params["rx_gain"], 
        Rate      = run_params["rate"], 
        decimation= True, 
        RF        = run_params["LO_freq"], 
        Front_end = run_params["front_end"],
        Device    = None, 
        Iterations= run_params["vna"]["iterations"], 
        verbose   = False,
        subfolder = None, 
        output_filename = outfname, 
        Multitone_compensation = N_power)
    print("Done.")

    ## Wait for the chip to cool off?
    print("Waiting for chip to cool...")
    time.sleep(cooltime_s)

    ## Fit the data acquired in this noise scan
    if (res_search_freqs_GHz is None): res_search_freqs_GHz = [ run_params["vna"]["f_center_Hz"]/1e9 ]
    print("Fitting VNA sweep to find resonator frequency...")
    fs, qs, _,_,_,_,_ = vna_file_fit(vna_filename + '.h5',res_search_freqs_GHz,show_plots=False,save=True,verbose=False)
    run_params["vna"]["result_fs"] = fs*1e9 ## Store it in Hz
    run_params["vna"]["result_qs"] = qs
    print("Done.")
    print("Fitted Fs (GHz):",fs)
    print("Fitted Qs      :",qs)

    ## Create a VNA group for our summary h5 file
    if h5_group_obj is not None:
        gVNA = h5_group_obj.create_group('VNA')
        gVNA.attrs.create("duration", run_params["delay"]["duration_s"])
        gVNA.attrs.create("n_points", run_params["vna"]["n_points"])
        gVNA.attrs.create("iteratns", run_params["vna"]["iterations"])
        gVNA.attrs.create("VNAfile",  outfname+".h5")

        ## Save the fit results to the VNA group
        gVNA.create_dataset('fit_f_GHz', data=np.array(fs))
        gVNA.create_dataset('fit_Q_fac', data=np.array(qs))

    ## Extract the important parameters from fit, Get res freqs in Hz (fs is in GHz)
    return fs*1e9, qs, vna_filename


def run_stream(series, run_params, h5_group_obj=None, cooltime_s=5, run_type="Noise", suffix=None):

    ## Create some output objects
    ## Each entry is a single number
    n_c_deltas = len(run_params["stream"]["cal_deltas"])
    cal_freqs = np.zeros(n_c_deltas)
    cal_means = np.zeros(n_c_deltas, dtype=np.complex_)

    ## Pull the VNA results
    f = run_params["vna"]["result_fs"][0]
    q = run_params["vna"]["result_qs"][0]

    ## Determine how many equivalent tones to split the DAC power output into to achieve the requested output power
    N_power = np.power(10.,(((-1*run_params["rf_power"])-14)/20.))
    pwr_clc = np.round(-14-20*np.log10(N_power),2)

    ## For each power, loop over all the calibration offsets
    for j, delta in enumerate(run_params["stream"]["cal_deltas"]):

        ## Make array of the central tone and tracking tones (e.g.: [fres, fTa, fTb])
        readout_tones  = np.append([f + delta*float(f)/q], run_params["stream"]["track_tones_Hz"])
        n_ro_tones     = len(readout_tones)
        readout_tones  = np.around(readout_tones, decimals=0)

        ## Split the power evenly across the tones
        amplitudes     = 1./N_power * np.ones(n_ro_tones)
        relative_tones = np.array([float(ro_tone) - run_params["LO_freq"] for ro_tone in readout_tones])

        ## Don't need tracking tones for calibration deltas
        if not (delta==0):
            relative_tones = np.array([relative_tones[0]])
            amplitudes     = np.array([amplitudes[0]])

        print("Readout  tones [Hz]:", readout_tones)
        print("Relative tones [Hz]:", relative_tones)
        print("Amplitudes:         ", amplitudes)
        print("LO Frequency [Hz]:  ", run_params["LO_freq"])

        outfname = "USRP_"+run_type+"_"+series+"_delta"+str(int(100.*delta))
        if (suffix is not None):
            outfname += "_" + str(suffix)

        ## Determine how long to acquire noise, passed in seconds
        dur_noise = ( run_params["stream"]["duration_s"] 
                    if ((np.abs(delta) < 0.005) and (run_params["stream"]["calib_s"] < run_params["stream"]["duration_s"])) 
                    else run_params["stream"]["calib_s"] )

        ## Create a group for the noise scan parameters
        if (h5_group_obj is not None) and (delta==0):
            gScan = h5_group_obj.create_group('Scan'+str(j))
            gScan.attrs.create("delta", delta)
            gScan.attrs.create("file",  outfname+".h5")
            gScan.attrs.create("LOfrequency", run_params["LO_freq"])
            gScan.attrs.create("duration",  dur_noise)
            gScan.create_dataset("readout_tones",  data=readout_tones)
            gScan.create_dataset("relative_tones", data=relative_tones)
            gScan.create_dataset("amplitudes",     data=amplitudes)
            
            gScan.attrs.create("timestart", time.time())

        print("Starting Noise Run...")
        ## Do a noise run with the USRP
        noise_file = u.get_tones_noise(relative_tones, 
                                    #measure_t  = lapse_noise,  ## passed in sec
                                    measure_t  = dur_noise,
                                    tx_gain    = run_params["tx_gain"], 
                                    rx_gain    = run_params["rx_gain"], 
                                    rate       = run_params["rate"],  ## passed in Samps per sec
                                    decimation = 100, 
                                    RF         = run_params["LO_freq"],  ## passed in Hz 
                                    Front_end  = run_params["front_end"],
                                    Device     = None,
                                    amplitudes = amplitudes,
                                    delay      = run_params["delay"]["value_s"], ## passed in sec
                                    pf_average = 4, 
                                    mode       = "DIRECT", 
                                    trigger    = None, 
                                    shared_lo  = False,
                                    subfolder  = None,
                                    output_filename = outfname)

        ## Wait for the chip to cool off?
        print("Waiting for chip to cool...")
        time.sleep(cooltime_s) ## 30 seconds

        ## Add an extension to the file path
        noise_file += '.h5'

        ## Do a quick average of the timestream
        time_threshold = 0.3 * dur_noise
        raw_ts, info   = read_timestream(noise_file)
        mean_ts        = avg_timestream(raw_ts, info, time_threshold=time_threshold, verbose=False)
        del raw_ts

        ## Store the result in the internal output arrays
        cal_freqs[j] = info['search freqs'][0] ## in MHz
        cal_means[j] = mean_ts[0]

        if not (delta == 0):
            os.remove(noise_file)

    return cal_freqs, cal_means, noise_file


def run_full_suite(series, run_params, f_res_GHz, run_type="Noise", h5_group_obj=None, subrun_id=0):

    ## Instantiate an output file
    new_file = False
    if h5_group_obj is None:
        new_file = True
        h5_group_obj = h5py.File('noise_averages_'+series+'.h5','w')

    ## Create an h5 group for this data, store some general metadata
    gSubrun = h5_group_obj.create_group('Run'+str(int(subrun_id)))
    gSubrun.attrs.create("power",   run_params["rf_power"])
    gSubrun.attrs.create("tx_gain", run_params["tx_gain"])
    gSubrun.attrs.create("rx_gain", run_params["rx_gain"])
    # gSubrun.attrs.create("N_power", N_power)
    gSubrun.attrs.create("rate",    run_params["rate"])
    gSubrun.attrs.create("LOfreq",  run_params["LO_freq"])

    _, _ = run_delay(series, run_params, h5_group_obj=gSubrun, delay_over_s=None)

    _, _, _ = run_vna(series, run_params, res_search_freqs_GHz=f_res_GHz, h5_group_obj=gSubrun, cooltime_s=5)

    cal_freqs, cal_means, _ = run_stream(series, run_params, h5_group_obj=gSubrun, cooltime_s=5, type=run_type)

    ## Store the resulting arrays in this h5 group
    gSubrun.create_dataset('freqs',data=cal_freqs)
    gSubrun.create_dataset('means',data=cal_means)

    ## Close h5 file for writing
    if new_file: h5_group_obj.close()