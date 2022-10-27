import sys,os
import time, datetime
import argparse
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
f_span_kHz = 140        ## Symmetric about the center frequency
points     = 1400       ## Defined such that we look at 100 Hz windows
duration   = 10         ## [Sec]

## Set some VNA parameters
power   = -30.0     ## [dBm]
                       
## File handling options
filename=None

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_VNA_Sweeps'  #VNA subfolder of TempSweeps

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
    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--power' , '-P' , type=float, default = power, 
        help='RF power applied in dBm. (default '+str(power)+' dBm)')
    parser.add_argument('--txgain', '-tx', type=float, default = tx_gain, 
        help='Tx gain factor (default '+str(tx_gain)+')')
    parser.add_argument('--rxgain', '-rx', type=float, default = rx_gain, 
        help='Rx gain factor (default '+str(rx_gain)+')')
    parser.add_argument('--rate'  , '-R' , type=float, default = rate/1e6, 
        help='Sampling frequency (default '+str(rate/1e6)+' Msps)')
    parser.add_argument('--points', '-p' , type=int  , default=points, 
        help='Number of points used in the scan (default '+str(points)+' points)')
    parser.add_argument('--time'  , '-T' , type=float, default=duration, 
        help='Duration of the scan in seconds per iteration (default '+str(duration)+' seconds)')
    

    parser.add_argument('--iter'  , '-i' , type=int, default=1, 
        help='How many iterations to perform (default 1)')
    
    parser.add_argument('--LOfrq' , '-f' , type=float, default=LO/1e6,
        help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default '+str(LO/1e6)+' MHz)')
    parser.add_argument('--VNAfspan', '-fv', type=float, default=f_span_kHz,
        help='Frequency span in kHz over which to do the VNA scan (default '+str(f_span_kHz)+' kHz)')

    args = parser.parse_args()

    # Do some conditional checks

    if (args.power is not None):
        print("Power(s):", args.power, type(args.power))

        min_pwer = -70.0
        max_pwer = -15.0

        if (args.power < min_pwer):
            print("Power",args.power,"too Low! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to minimum...")
            args.power = min_pwer

        if (args.power > max_pwer):
            print("Power",args.power,"too High! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to maximum...")
            args.power = max_pwer

    if (args.rate is not None):
        args.rate = args.rate * 1e6 ## Store it as sps not Msps
        if (args.rate > rate):
            print("Rate",args.rate,"is too High! Optimal performance is at",rate,"samples per second")
            args.rate = rate

    if (args.iter is not None):
        if (args.iter < 0):
            args.iter = 1

    ## MHz frequencies to Hz
    if (args.LOfrq is not None):
        args.LOfrq = args.LOfrq*1e6 ## Store it as Hz not MHz
    if (args.VNAfspan is not None):
        args.VNAfspan = args.VNAfspan*1e3 ## Store it as Hz not kHz
        if(args.VNAfspan > 1e7):
            print("Frequency range (",args.VNAfspan,") too large! Exiting...")
            exit(1)

    if(args.LOfrq is not None):
        if(np.any(np.array(args.LOfrq) > 6e9)):
            print("Invalid LO Frequency:",args.freq," is too High! Exiting...")
            exit(1)

    # if np.abs(args.f0)>args.rate/2:
    #     u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f0/1e6,args.rate/1e6))
    #     args.f0 = args.rate/2 * (np.abs(args.f0)/args.f0)
    #     u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(args.f0/1e6))

    # if np.abs(args.f1)>args.rate/2:
    #     u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f1/1e6,args.rate/1e6))
    #     args.f1 = args.rate/2 * (np.abs(args.f1)/args.f1)
    #     u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(args.f1/1e6))

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

## Function to run a VNA sweep
def runVNA(tx_gain, rx_gain, _iter, rate, freq, front_end, fspan, lapse, points, ntones, delay_duration, delay_over=None):
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

    if ntones ==1:
        ntones = None
    print("Using", ntones, "tones for Multitone_compensation")

    outfname = "USRP_VNA_"+series

    # print("VNA arguments:")
    # print("-    start_f", f0)
    # print("-     last_f", f1)
    # print("-  measure_t", lapse)
    # print("-   n_points", points)
    # print("-    tx_gain", tx_gain)
    # print("-    rx_gain", rx_gain)
    # print("-       Rate", rate)
    # print("- decimation", True)
    # print("-         RF", freq)
    # print("-  Front_end", front_end)
    # print("-     Device", None)
    # print("- Iterations", _iter)
    # print("-    verbose", False)
    # print("-  subfolder", seriesPath)
    # print("- output_filename", outfname)
    # print("- Multitone_compensation", ntones)

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
        measure_t = lapse, 
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
        subfolder = seriesPath,
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

    return vna_filename, delay

## Main function when called from command line
if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    ## Create the output directories
    dateStr, sweepPath, series, seriesPath = get_paths()
    create_dirs()
    os.chdir(seriesPath) ## When doing this, no need to provide subfolder

    ## Ensure the power doesn't go above -25 dBm
    ## Due to power splitting across tones
    if power > -25:
        USRP_power   = -25
        args.txgain = power - USRP_power
    else:
        USRP_power   = power

    ## Calculate some powers
    N_power = np.power(10.,(((-1*args.power)-14)/20.))
    pwr_clc = np.round(-14-20.*np.log10(N_power),2)

    print(pwr_clc, 'dBm of power')
    print(N_power, 'is the equivalent number of tones needed to split the DAQ power into the above amount')

    ## Print the settings we'll be using
    print('===== VNA Settings =====')
    print('     LO [MHz]: ',args.LOfrq/1e6)
    print('     f0 [MHz]: ',args.f0/1e6)
    print('     f1 [MHz]: ',args.f1/1e6)
    print('  power [dBm]: ',args.power)
    print('  rate [Msps]: ',args.rate/1e6)
    print('      Tx gain: ',args.txgain)
    print('      Rx gain: ',args.rxgain)
    print('  N of points: ',args.points)
    print('   Iterations: ',args.iter)
    print(' Duration (s): ',args.time)

    # Data acquisition
    for fi in range(len(args.LOfrq)):
        vna_file, delay = runVNA(
            tx_gain = args.txgain,
            rx_gain = args.rxgain,
            _iter   = args.iter,
            rate    = args.rate,        ## Passed in Samps/sec
            freq    = args.LOfrq[fi],   ## Passed in Hz
            front_end = "A",
            fspan   = args.VNAfspan,      ## Passed in Hz
            # f0      = args.f0,          ## Passed in Hz, relative to LO
            # f1      = args.f1,          ## Passed in Hz, relative to LO
            lapse   = args.time,        ## Passed in seconds
            points  = args.points,
            ntones  = N_power,
            delay_duration = 0.1, # args.delay_duration,
            delay_over = None) #args.delay_over)

    u.Disconnect()