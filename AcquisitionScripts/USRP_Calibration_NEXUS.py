## Import the relevant modules
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
        exit()

## Set some VNA parameters
power   = -20.0     ## [dBm]
rate    = 100e6     ## samples per second
tx_gain =  0
rx_gain = 17.5
LO      =  4.250e9  ## [Hz] Nice round numbers, don't go finer than 50 MHz
f0      = -1.000e7  ## [Hz], relative to LO
f1      = -0.000e7  ## [Hz], relative to LO
points  =  1.00e5
lapse   = 10        ## [Sec]
                       
## File handling options
filename=None

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_Calibrations'  #VNA subfolder of TempSweeps

## Sub directory definitions
dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath = os.path.join(dataPath,dateStr)

series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = os.path.join(sweepPath,series)

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
    parser.add_argument('--time'  , '-T' , type=float, default=lapse, 
        help='Duration of the scan in seconds per iteration (default '+str(lapse)+' seconds)')
    

    parser.add_argument('--iter'  , '-i' , type=int, default=1, 
        help='How many iterations to perform (default 1)')
    
    parser.add_argument('--LOfrq' , '-f' , nargs='+' , default=[LO/1e6],
        help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default '+str(LO/1e6)+' MHz)')
    parser.add_argument('--f0'    , '-f0', type=float, default=f0/1e6, 
        help='Baseband start frequrency in MHz relative to LO (default '+str(f0/1e6)+' MHz)')
    parser.add_argument('--f1'    , '-f1', type=float, default=f1/1e6, 
        help='Baseband end frequrency in MHz relative to LO (default '+str(f1/1e6)+' MHz)')

    args = parser.parse_args()

    # Do some conditional checks

    if (args.power is not None):
        print("Power(s):", args.power, type(args.power))

        min_pwer = -70.0
        max_pwer = -15.0

        if (args.power < min_pwer):
            print("Power",args.power,"too Low! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to minimum...")
            args.power = min_pwer

        # Don't need to enforce this because it is used to tune up the tx gain later
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
        args.LOfrq = [f*1e6 for f in args.LOfrq] ## Store it as Hz not MHz
    if (args.f0 is not None):
        args.f0 = args.f0*1e6 ## Store it as Hz not MHz
    if (args.f1 is not None):
        args.f1 = args.f1*1e6 ## Store it as Hz not MHz

    if(args.f0 is not None and args.f1 is not None):
        if((args.f1 - args.f0) > 1e7):
            print("Frequency range (",args.f0,",",args.f1,") too large! Exiting...")
            exit(1)

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

## Function to run a VNA sweep
def runVNA(tx_gain, rx_gain, _iter, rate, freq, front_end, f0, f1, lapse, points, ntones, delay_duration, delay_over=None):

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
                subfolder=seriesPath)
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

    outfname = "USRP_VNA_"+series

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

    return vna_filename, delay

## Main function when called from command line
if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Create the output directories
    create_dirs()
    os.chdir(seriesPath)

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    ## Calculate some powers
    N_power = np.power(10.,(((-1*args.power)-14)/20.))
    pwr_clc = np.round(-14-20.*np.log10(N_power),2)

    print(pwr_clc, 'dBm of power')
    print(N_power, 'is the equivalent number of tones needed to split the DAQ power into the above amount')

    ## Print the settings we'll be using
    print('===== VNA Settings =====')
    print('     LO [MHz]: ',[f/1e6 for f in args.LOfrq])
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
            f0      = args.f0,          ## Passed in Hz, relative to LO
            f1      = args.f1,          ## Passed in Hz, relative to LO
            lapse   = args.time,        ## Passed in seconds
            points  = args.points,
            ntones  = N_power,
            delay_duration = 0.1, # args.delay_duration,
            delay_over = None) #args.delay_over)

    u.Disconnect()