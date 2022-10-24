## To Do:
## - Implement arguments for tracking tones, etc

## Import the relevant modules
import sys, os
import time, datetime
import argparse
import numpy as np

import h5py

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

## Set DAQ parameters
rate    = 100e6
tx_gain = 0
rx_gain = 17.0
LO      = 4.25e9       ## (Al and Nb 7) [Hz] Round numbers, no finer than 50 MHz
# LO      = 4.20e9       ## (Nb 6) [Hz] Round numbers, no finer than 50 MHz

## Set Resonator parameters
res     = 4.24204767      ## Al   [GHz]
# res     = 4.244760      ## Nb 7 [GHz]
# res     = 4.202830      ## Nb 6 [GHz]

## Set some VNA sweep parameters
f0      = -10e6         ## (Al and Nb 7) [Hz], relative to LO=4.25e9
f1      = -5e6          ## (Al and Nb 7) [Hz], relative to LO=4.25e9
# f0      = -5e6          ## (Nb 6) [Hz], relative to LO=4.20e9
# f1      =  5e6          ## (Nb 6) [Hz], relative to LO=4.20e9
f_span_kHz = 200        ## Symmetric about the center frequency
points     = 2e5
duration   = 30         ## [Sec]

## Set the non-resonator tracking tones
tracking_tones = np.array([4.235e9,4.255e9]) ## (Al or Nb 7)    In Hz a.k.a. cleaning tones to remove correlated noise
# tracking_tones = np.array([4.193e9,4.213e9]) ## (Nb 6)  In Hz a.k.a. cleaning tones to remove correlated noise

## Set the stimulus powers to loop over
powers = np.array([-30])
# powers = np.array([-60,-55,-50,-45,-40,-35,-30])
# powers = np.arange(start=-60,stop=-10,step=5)
# powers = np.array([-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15])
# powers = np.array([-50,-45,-40,-35,-30,-25,-20,-15])
n_pwrs = len(powers)

## Set the deltas to scan over in calibrations
## These deltas are fractions of the central frequency
## This can be used to do a pseudo-VNA post facto
cal_deltas = np.linspace(start=-0.05, stop=0.05, num=3)
n_c_deltas = len(cal_deltas)
cal_lapse_sec = 10.

## File handling options
filename=None

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_Noise_Scans'

## Sub directory definitions
dateStr    = '' # str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath  = '' # os.path.join(dataPath,dateStr)

series     = '' # str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = '' # os.path.join(sweepPath,series)

def get_paths():
    ## Sub directory definitions
    dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
    sweepPath = os.path.join(dataPath,dateStr)

    series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    seriesPath = os.path.join(sweepPath,series)

    return dateStr, sweepPath, series, seriesPath

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Acquire a noise timestream with the USRP using the GPU_SDR backend.')

    parser.add_argument('--power'    , '-P' , type=float, default = powers[0], 
        help='RF power applied in dBm. (default '+str(powers[0])+' dBm)')
    parser.add_argument('--txgain'   , '-tx', type=float, default = tx_gain, 
        help='Tx gain factor (default '+str(tx_gain)+')')
    parser.add_argument('--rxgain'   , '-rx', type=float, default = rx_gain, 
        help='Rx gain factor (default '+str(rx_gain)+')')
    parser.add_argument('--rate'     , '-R' , type=float, default = rate/1e6, 
        help='Sampling frequency (default '+str(rate/1e6)+' Msps)')
    parser.add_argument('--points'   , '-p' , type=int  , default=points, 
        help='Number of points used in the scan (default '+str(points)+' points)')
    parser.add_argument('--timeVNA'  , '-Tv' , type=float, default=duration, 
        help='Duration of the VNA scan in seconds per iteration (default '+str(duration)+' seconds)')
    parser.add_argument('--timeNoise', '-Tn' , type=float, default=duration, 
        help='Duration of the noise scan in seconds (default '+str(duration)+' seconds)')

    parser.add_argument('--iter'  , '-i' , type=int, default=1, 
        help='How many iterations to perform (default 1)')
    
    parser.add_argument('--LOfrq' , '-f' , type=float, default=LO/1e6,
        help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default '+str(LO/1e6)+' MHz)')
    parser.add_argument('--VNAfspan', '-fv', type=float, default=f_span_kHz/1e3,
        help='Frequency span in kHz over which to do the VNA scan (default '+str(f_span_kHz/1e3)+' MHz')
    # parser.add_argument('--f0'    , '-f0', type=float, default=f0/1e6, 
    #     help='Baseband start frequency in MHz, absolute (default '+str(f0/1e6)+' MHz)')
    # parser.add_argument('--f1'    , '-f1', type=float, default=f1/1e6, 
    #     help='Baseband end frequency in MHz, absolute (default '+str(f1/1e6)+' MHz)')

    args = parser.parse_args()

    ## Do some conditional checks

    if (args.power is not None):
        print("Power(s):", args.power, type(args.power))

        powers[0] = args.power
        n_pwrs = len(powers)

        min_pwer = -70.0
        max_pwer = -15.0
        for i in np.arange(n_pwrs):
            if (powers[i] < min_pwer):
                print("Power",args.power,"too Low! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to minimum...")
                powers[i] = min_pwer

            if (powers[i] > max_pwer):
                print("Power",args.power,"too High! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to maximum...")
                powers[i] = max_pwer

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
    # if (args.f0 is not None):
    #     args.f0 = args.f0*1e6 ## Store it as Hz not MHz
    # if (args.f1 is not None):
    #     args.f1 = args.f1*1e6 ## Store it as Hz not MHz

    # if(args.f0 is not None and args.f1 is not None):
    #     if((args.f1 - args.f0) > 1e7):
    #         print("Frequency range (",args.f0,",",args.f1,") too large! Exiting...")
    #         exit(1)

    if(args.LOfrq is not None):
        if(np.any(np.array(args.LOfrq) > 6e9)):
            print("Invalid LO Frequency:",args.freq," is too High! Exiting...")
            exit(1)

    if np.abs(args.f0)>args.rate/2:
        u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f0/1e6,args.rate/1e6))
        args.f0 = args.rate/2 * (np.abs(args.f0)/args.f0)
        u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(args.f0/1e6))

    if np.abs(args.f1)>args.rate/2:
        u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f1/1e6,args.rate/1e6))
        args.f1 = args.rate/2 * (np.abs(args.f1)/args.f1)
        u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(args.f1/1e6))

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

def runNoise(tx_gain, rx_gain, _iter, rate, freq, front_end, fspan, lapse_VNA, lapse_noise, points, ntones, delay_duration, delay_over=None, h5_group_obj=None):
    ## First do a line delay measurement
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
                duration = delay_duration,
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
    fVNAmin = res*1e9 - (fspan/2.)
    fVNAmax = res*1e9 + (fspan/2.)
    print("VNA spans", fVNAmin/1e6, "MHz to", fVNAmax/1e6, "MHz")
    f0 = freq - fVNAmin
    f1 = freq - fVNAmax
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

    ## Fit the data acquired in this noise scan
    print("Fitting VNA sweep to find resonator frequency...")
    fs, qs, _,_,_,_,_ = puf.vna_file_fit(vna_filename + '.h5',[res],show=True,save=True)
    print("Done.")

    ## Extract the important parameters from fit
    f = fs[0]*1e9 ## Get it in Hz (fs is in GHz)
    q = qs[0]
    print("F:",f,"Q:",q)

    ## Save the fit results to the VNA group
    gVNA.create_dataset('fit_f_GHz', data=np.array(fs[0]))
    gVNA.create_dataset('fit_Q_fac', data=np.array(qs[0]))

    ## Create some output objects
    ## Each entry is a single number
    cal_freqs = np.zeros(n_c_deltas)
    cal_means = np.zeros(n_c_deltas, dtype=np.complex_)

    ## For each power, loop over all the calibration offsets
    for j in np.arange(n_c_deltas):
        ## Pick this calibration delta
        delta = cal_deltas[j]

        ## Make array of tones (fres, fTa, fTb)
        readout_tones  = np.append([f + delta*float(f)/q], tracking_tones)
        n_ro_tones     = len(readout_tones)
        readout_tones  = np.around(readout_tones, decimals=0)

        ## Split the power evenly across the tones
        amplitudes     = 1./ntones * np.ones(n_ro_tones)

        relative_tones = np.zeros(n_ro_tones)
        for k in np.arange(n_ro_tones):
            relative_tones[k] = float(readout_tones[k]) - freq

        ## Don't need tracking tones for calibration deltas
        if not (delta==0):
            relative_tones = np.array([relative_tones[0]])
            amplitudes     = np.array([amplitudes[0]])

        outfname = "USRP_Noise_"+series+"_delta"+str(int(100.*delta))

        ## Create a group for the noise scan parameters
        gScan = h5_group_obj.create_group('Scan'+str(j))
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

def doRun(this_power):

    ## Create the output directories
    create_dirs()
    os.chdir(seriesPath) ## When doing this, no need to provide subfolder

    ## Instantiate an output file
    fyle = h5py.File(os.path.join(seriesPath,'noise_averages_'+series+'.h5'),'w')

    ## Ensure the power doesn't go above -25 dBm
    ## Due to power splitting across tones
    if this_power > -25:
        USRP_power   = -25
        args.txgain = this_power - USRP_power
    else:
        USRP_power   = this_power

    ## Calculate some derived quantities
    N_power = np.power(10.,(((-1*USRP_power)-14)/20.))
    pwr_clc = np.round(-14-20*np.log10(N_power),2)

    print("Initializing Noise Scan...")
    print(pwr_clc, 'dBm of power')
    print(N_power, 'is the equivalent number of tones needed to split the DAQ power into the above amount')

    ## Create an h5 group for this data, store some general metadata
    gPower = fyle.create_group('Power'+str(0)) #+str(i))
    gPower.attrs.create("power",   USRP_power)
    gPower.attrs.create("tx_gain", args.txgain)
    gPower.attrs.create("rx_gain", args.rxgain)
    gPower.attrs.create("N_power", N_power)
    gPower.attrs.create("rate",    args.rate)
    gPower.attrs.create("LOfreq",  args.LOfrq)

    cal_freqs, cal_means = runNoise(
        tx_gain = args.txgain,
        rx_gain = args.rxgain,
        _iter   = args.iter,
        rate    = args.rate,        ## Passed in Samps/sec
        freq    = args.LOfrq,       ## Passed in Hz
        front_end = "A",
        fspan   = args.VNAfspan,    ## Passed in Hz
        # f0      = args.f0,          ## Passed in Hz, relative to LO
        # f1      = args.f1,          ## Passed in Hz, relative to LO
        lapse_VNA   = args.timeVNA,   ## Passed in seconds
        lapse_noise = args.timeNoise, ## Passed in seconds
        points  = args.points,
        ntones  = N_power,
        delay_duration = 0.1, # args.delay_duration,
        delay_over = None,
        h5_group_obj = gPower) #args.delay_over)

    ## Store the resulting arrays in this h5 group
    gPower.create_dataset('freqs',data=cal_freqs)
    gPower.create_dataset('means',data=cal_means)

    ## Close h5 file for writing
    fyle.close()

if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    ## Loop over the powers considered
    for i in np.arange(n_pwrs):
        dateStr, sweepPath, series, seriesPath = get_paths()

        doRun(powers[i])

    ## Disconnect from the USRP server
    u.Disconnect()
