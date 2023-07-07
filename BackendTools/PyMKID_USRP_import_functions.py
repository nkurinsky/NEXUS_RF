import sys, os
import time, datetime
import h5py

import numpy as np

## Try to read in the USRP modules
## Exit out if you can't after adjusting path
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../DeviceControl/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")
        exit()

try:
    import PyMKID_USRP_functions as puf
except ImportError:
    try:
        sys.path.append('../AnalysisScripts')
        import PyMKID_USRP_functions as puf
    except ImportError:
        print("Cannot find the PyMKID_USRP_functions package")
        exit()


def run_delay(series, tx_gain, rx_gain, rate, freq, front_end, lapse_delay, delay_over=None, h5_group_obj=None):

    if delay_over is not None:
        print("Line delay is user specified:", delay_over, "ns")
        delay = delay_over
        u.set_line_delay(rate, delay_over*1e9)
    else:
        try:
            if u.LINE_DELAY[str(int(rate/1e6))]:
                delay = u.LINE_DELAY[str(int(rate/1e6))]*1e-9
                print("Line delay was found in file.")
        except KeyError:
            print("Cannot find line delay. Measuring line delay before VNA:")

            outfname = "USRP_Delay_"+series

            filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, 
                tx_gain=tx_gain, 
                rx_gain=rx_gain, 
                compensate = True, 
                duration = lapse_delay,
                output_filename=outfname, 
                subfolder=None)#seriesPath)
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
    h5_group_obj.attrs.create("delay_ns", delay * 1e9)

    return delay

def run_vna(series, res, tx_gain, rx_gain, _iter, rate, freq, front_end, fspan, lapse_VNA, points, ntones, h5_group_obj=None):

    if ntones ==1:
        ntones = None
    print("Using", ntones, "tones for Multitone_compensation")

    outfname = "USRP_VNA_"+series

    ## Create a VNA group for our h5 file
    gVNA = h5_group_obj.create_group('VNA')
    gVNA.attrs.create("duration", lapse_VNA)
    gVNA.attrs.create("n_points", points)
    gVNA.attrs.create("iteratns", _iter)
    gVNA.attrs.create("VNAfile",  outfname+".h5")

    ## Do some math to find the frequency span for the VNA
    ## relative to the LO frequency
    print("F span (VNA):",fspan,"Hz")
    fVNAmin = res*1e9 - (fspan/2.)
    fVNAmax = res*1e9 + (fspan/2.)
    print("VNA spans", fVNAmin/1e6, "MHz to", fVNAmax/1e6, "MHz")
    f0 = fVNAmin - freq
    f1 = fVNAmax - freq
    print("Relative to LO: start", f0, "Hz; stop",f1,"Hz")

    print("Starting single VNA run...")
    vna_filename  = u.Single_VNA(start_f = f0, last_f = f1, 
        measure_t = lapse_VNA, 
        n_points  = points, 
        tx_gain   = tx_gain,
        rx_gain   = rx_gain, 
        Rate      = rate, 
        decimation= True, 
        RF        = freq, 
        Front_end = front_end,
        Device    = None, 
        Iterations= _iter, 
        verbose   = False,
        subfolder = None, #seriesPath,
        output_filename = outfname, 
        Multitone_compensation = ntones)
    print("Done.")

    ## Wait for the chip to cool off?
    print("Waiting for chip to cool...")
    time.sleep(5) ## 30 seconds

    ## Fit the data acquired in this noise scan
    print("Fitting VNA sweep to find resonator frequency...")
    fs, qs, _,_,_,_,_ = puf.vna_file_fit(vna_filename + '.h5',[res],show=False)
    print("Done.")

    ## Extract the important parameters from fit
    f = fs[0]*1e9 ## Get it in Hz (fs is in GHz)
    q = qs[0]
    print("F:",f,"Q:",q)

    ## Save the fit results to the VNA group
    gVNA.create_dataset('fit_f_GHz', data=np.array(fs[0]))
    gVNA.create_dataset('fit_Q_fac', data=np.array(qs[0]))

    return f, q

def run_noise(series, delay, f, q, cal_deltas, tracking_tones, tx_gain, rx_gain, rate, freq, front_end, lapse_noise, cal_lapse_sec, points, ntones, h5_group_obj=None, idx=None):
    ## Create some output objects
    ## Each entry is a single number
    n_c_deltas = len(cal_deltas)
    cal_freqs = np.zeros(n_c_deltas)
    cal_means = np.zeros(n_c_deltas, dtype=np.complex_)

    ## For each power, loop over all the calibration offsets and take a noise run
    for j in np.arange(n_c_deltas):
        ## Pick this calibration delta
        delta = cal_deltas[j]

        ## Make array of tones (fres, fTa, fTb)
        readout_tones  = np.append([f + delta*float(f)/q], tracking_tones)
        n_ro_tones     = len(readout_tones)
        readout_tones  = np.around(readout_tones, decimals=0)

        ## Split the power evenly across two tones
        amplitudes     = 1./ntones * np.ones(n_ro_tones)

        relative_tones = np.zeros(n_ro_tones)
        for k in np.arange(n_ro_tones):
            relative_tones[k] = float(readout_tones[k]) - freq

        ## Don't need tracking tones for calibration deltas
        if not (delta==0):
            relative_tones = np.array([relative_tones[0]])
            amplitudes     = np.array([amplitudes[0]])

        outfname = "USRP_Noise_"+series+"_delta"+str(int(100.*delta))
        if idx is not None:
            outfname += "_" + str(idx)

        ## Create a group for the noise scan parameters
        gScan = h5_group_obj.create_group('Scan'+str(j) + (('_'+str(idx)) if idx is not None else ''))
        gScan.attrs.create("delta", delta)
        gScan.attrs.create("file",  outfname+".h5")
        gScan.create_dataset("readout_tones",  data=readout_tones)
        gScan.create_dataset("relative_tones", data=relative_tones)
        gScan.create_dataset("amplitudes",     data=amplitudes)
        gScan.create_dataset("LOfrequency",    data=np.array([freq]))

        print("Readout  tones [Hz]:", readout_tones)
        print("Relative tones [Hz]:", relative_tones)
        print("Amplitudes:         ", amplitudes)
        print("LO Frequency [Hz]:  ", freq)

        ## Determine how long to acquire noise
        dur_noise = lapse_noise if ((np.abs(delta) < 0.005) and (cal_lapse_sec < lapse_noise)) else cal_lapse_sec  ## passed in sec
        gScan.create_dataset("duration",       data=np.array([dur_noise]))

        print("Starting Noise Run...")
        ## Record the start time
        start_time = datetime.datetime.now()
        start_tstr = str(start_time.strftime('%Y%m%d_%H%M%S'))

        ## Do a noise run with the USRP
        noise_file = u.get_tones_noise(relative_tones, 
                                    #measure_t  = lapse_noise,  ## passed in sec
                                    measure_t  = dur_noise,
                                    tx_gain    = tx_gain, 
                                    rx_gain    = rx_gain, 
                                    rate       = rate,  ## passed in Hz
                                    decimation = 100, 
                                    RF         = freq,  ## passed in Hz 
                                    Front_end  = front_end,
                                    Device     = None,
                                    amplitudes = amplitudes,
                                    delay      = delay, ## passed in ns
                                    pf_average = 4, 
                                    mode       = "DIRECT", 
                                    trigger    = None, 
                                    shared_lo  = False,
                                    subfolder  = None,#seriesPath,
                                    output_filename = outfname)

        ## Save the start time to the h5 data object
        gScan.create_dataset("start_time",   data=np.array([start_time]))
        gScan.create_dataset("start_string", data=np.array([start_tstr]))

        ## Wait for the chip to cool off?
        print("Waiting for chip to cool...")
        time.sleep(5) ## 30 seconds

        ## Add an extension to the file path
        noise_file += '.h5'

        time_threshold = 0.3 * dur_noise # dur_noise / 2

        frequencies_scanned, noise_mean_scanned = puf.avg_noi(noise_file,time_threshold=time_threshold)

        ## Store the result in the internal output arrays
        cal_freqs[j] = frequencies_scanned[0]
        cal_means[j] = noise_mean_scanned[0]

        if not (delta == 0):
            os.remove(noise_file)

    return cal_freqs, cal_means