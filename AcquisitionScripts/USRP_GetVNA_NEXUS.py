import sys,os
import time, datetime
import argparse
import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../DeviceControl/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

import PyMKID_USRP_import_functions as puf2

## Default VNA sweep options 
power  = -50.0      ## in dBm
freq   = 4.24e3     ## in MHz
rate   = 200.0      ## in Msamps/sec

f0     = 1.0        ## in MHz
f1     = 4.0        ## in MHz

points = 1e3

lapse  = 20         ## in Seconds

delay_duration=0.01
delay_over='null'
                       
## File handling options
filename=None

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_VNA_Sweeps'  #VNA subfolder of TempSweeps

## Sub directory definitions
dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath = dataPath + '/' + dateStr

series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = sweepPath + '/' + series 

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Test the basic VNA functionality.')

    parser.add_argument('--power'   , '-P'    , type=float, default = power, 
        help='RF power applied in dBm. (default '+str(power)+' dBm)')
    parser.add_argument('--freq'    , '-f'    , nargs='+' , 
        help='LO frequency in MHz. Specifying multiple RF frequencies results in multiple scans (per each gain) (default '+str(freq)+' MHz)')
    parser.add_argument('--rate'    , '-r'    , type=float, default = rate, 
        help='Sampling frequency in Msps (default '+str(rate)+' Msps)')
    parser.add_argument('--frontend', '-rf'   , type=str  , default="A", 
        help='front-end character: A or B (default A)')
    parser.add_argument('--f0'      , '-f0'   , type=float, default=f0, 
        help='Baseband start frequrency in MHz (default '+str(f0)+' MHz)')
    parser.add_argument('--f1'      , '-f1'   , type=float, default=f1, 
        help='Baseband end frequrency in MHz (default '+str(f1)+' MHz)')
    parser.add_argument('--points'  , '-p'    , type=float, default=points, 
        help='Number of points used in the scan (default '+str(points)+' points)')
    parser.add_argument('--time'    , '-t'    , type=float, default=lapse, 
        help='Duration of the scan in seconds per iteration (default '+str(lapse)+' seconds)')
    parser.add_argument('--iter'    , '-i'    , type=float, default=1, 
        help='How many iterations to perform (default 1)')
    parser.add_argument('--gain'    , '-g'    , nargs='+' , 
        help='set the transmission gain. Multiple gains will result in multiple scans (per frequency) (default 0 dB)')

    ## Line delay arguments
    parser.add_argument('--delay_duration', '-dd', type=float, default=delay_duration, 
        help='Duration of the delay measurement (default '+str(delay_duration)+' seconds)')
    parser.add_argument('--delay_over'    , '-do', type=float, 
        help='Manually set line delay in nanoseconds. Skip the line delay measure.')

    args = parser.parse_args()

    # Do some conditional checks
    if(args.f0 is not None and args.f1 is not None):
        if((args.f1 - args.f0) > 1e7):
            print("Frequency range (",args.f0,",",args.f1,") too large")
            exit(1)

    if(args.freq is not None):
        if(np.any(args.freq > 6e9)):
            print("Invalid LO Frequency:",args.freq)
            exit(1)

    if(args.power is not None):
        if(args.power < -70):
            print("Power",args.power,"too Low! Range is -70 to 0 dBm")
            exit(1)

        elif(args.power > 0):
            print("Power",args.power,"too High! Range is -70 to 0 dBm")
            exit(1)

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
def runVNA(tx_gain, rx_gain, iter, rate, freq, front_end, f0, f1, lapse, points, ntones, delay_duration, delay_over):

    try:
        delay = u.LINE_DELAY[str(int(rate/1e6))]
        print("Line delay was found in file.")
    except KeyError:

        if delay_over is None:
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

        else:
            print("Line delay is user specified:", delay_over)
            delay = delay_over

            u.set_line_delay(rate, delay_over)

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
        Iterations= iter, 
        verbose   = False,
        subfolder = None,
        output_filename = None, 
        Multitone_compensation = ntones)
    print("Done.")

    return vna_filename, delay

## Main function when called from command line
if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    if args.freq is None:
        frequencies = [freq,]
    else:
        frequencies = [float(a) for a in args.freq]

    if args.gain is None:
        gains = [0,]
    else:
        gains = [int(float(a)) for a in args.gain]

    ## Create the output directories
    create_dirs()
    os.chdir(seriesPath)

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    if np.abs(args.f0)>args.rate/2:
        u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f0,args.rate))
        f0 = args.rate/2 * (np.abs(args.f0)/args.f0)
        u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(f0))
    else:
        f0 = args.f0

    if np.abs(args.f1)>args.rate/2:
        u.print_warning("Cannot use initial baseband frequency of %.2f MHz with a data rate of %.2f MHz" % (args.f1,args.rate))
        f1 = args.rate/2 * (np.abs(args.f1)/args.f1)
        u.print_debug("Setting maximum initial baseband scan frequency to %.2f MHz"%(f1))
    else:
        f1 = args.f1

    ## Calculate some powers
    N_power = 10**(((-1*args.power)-14)/20)

    print(np.round(-14-20*np.log10(N_power),2), 'dBm of power')
    print(N_power, 'is the equivalent number of tones needed to split the DAQ power into the above amount')

    ## Print the settings we'll be using
    print('===== VNA Settings =====')
    print('    LO [MHz]: ',frequencies)
    print('    f0 [MHz]: ',args.f0)
    print('    f1 [MHz]: ',args.f1)
    print(' power [dBm]: ',args.power)
    print(' tx+rx gains: ',gains)
    print('     npoints: ',args.points)

    # Data acquisition
    for g in gains:
        for f in frequencies:
            vna_file, delay = runVNA(
                tx_gain = g,
                rx_gain = g,
                iter = int(args.iter),
                rate = args.rate*1e6,        ## Passed in Samps/sec
                freq = f*1e6,                ## Passed in Hz
                front_end = args.frontend,
                f0 = f0*1e6,                 ## Passed in Hz
                f1 = f1*1e6,                 ## Passed in Hz
                lapse = args.time,
                points = args.points,
                ntones = N_power,
                delay_duration = args.delay_duration,
                delay_over = args.delay_over)

    u.Disconnect()