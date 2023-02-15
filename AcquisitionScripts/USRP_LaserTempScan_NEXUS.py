## Import the relevant modules
import sys, os
import time, datetime
import argparse
import numpy as np

import h5py

## Try to read in the necessary modules
## Exit out if you can't after adjusting path
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")
try:
    from NEXUSFunctions import * #control NEXUS fridge
    from E3631A  import *
    from AFG3102 import *
except ImportError:
    print("Cannot find the NEXUSFunctions package")
    exit()
 
sys.path.insert(1,'../DeviceControl/GPU_SDR')
try: 
    import pyUSRP as u
except ImportError:
    print("Cannot find the pyUSRP package")
    exit()

sys.path.insert(1, "../BackendTools")
try:
    import PyMKID_USRP_functions as puf
except ImportError:
    print("Cannot find the PyMKID_USRP_functions package")
    exit()

## Temperature scan settings [K]
Temp_base =  15e-3
Temp_min  =  25e-3
Temp_max  = 350e-3
Temp_step =  25e-3

## Temperature stabilization params
tempTolerance =   1e-4       ## K
sleepTime     =  30.0        ## sec
stableTime    =  60.0        ## sec

## Create the temperature array
Temps = np.arange(Temp_min,Temp_max+Temp_step,Temp_step)

## Use this if starting at the top temperature
# Temps = Temps[::-1] 
# if (Temp_base) < Temps[-1]:
#     Temps = np.append(Temps,Temp_base)

# ## Use this if starting at base temperature
if (Temp_base) < Temps[0]:
   Temps = np.append(Temp_base,Temps)

## Set Laser parameters
afg_pulse_params = {
    "f_Hz" :  10.0,
    "pw_us":   1.0,
    "V_hi" :   5.0,
    "V_lo" :   0.0,
    "d_ms" :   5.0,
}
LED_voltages = np.arange(start=2.000, stop=6.000, step=1.000)
# LED_voltages = LED_voltages[::-1]

## Set DAQ parameters
rate    = 100e6
tx_gain = 0
rx_gain = 17.0
LO      = 4.25e9       ## (Al and Nb 7) [Hz] Round numbers, no finer than 50 MHz
# LO      = 4.20e9       ## (Nb 6) [Hz] Round numbers, no finer than 50 MHz
led_dec   = 100        ## Default decimation for the LED timestreams

## Set Resonator parameters
res     = 4.24204767      ## Al   [GHz]
# res     = 4.244760      ## Nb 7 [GHz]
# res     = 4.202830      ## Nb 6 [GHz]

## Set some VNA sweep parameters
f_span_kHz = 140        ## Symmetric about the center frequency
points     = 1400       ## Defined such that we look at 100 Hz windows
duration   = 10         ## [Sec]

## Set the non-resonator tracking tones
tracking_tones = np.array([4.235e9,4.255e9]) ## (Al)    In Hz a.k.a. cleaning tones to remove correlated noise
# tracking_tones = np.array([4.193e9,4.213e9]) ## (Nb 6)  In Hz a.k.a. cleaning tones to remove correlated noise

## Set the stimulus powers to loop over
powers  = np.array([-40,-35,-30])
n_pwrs  = len(powers)

## Set the deltas to scan over in calibrations
## These deltas are fractions of the central frequency
## This can be used to do a pseudo-VNA post facto
cal_deltas = np.linspace(start=-0.05, stop=0.05, num=3)
n_c_deltas = len(cal_deltas)
cal_lapse_sec = 10.

## Set the duration to wait after an LED timestream before doing the next USRP operation
sleep_sec = 15.0 # 30.0

## File handling options
filename=None

## Where to save the output data (hdf5 files)
dataPath = '/data/USRP_Laser_TempScan_Data'

## Sub directory definitions
dateStr    = '' # str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath  = '' # os.path.join(dataPath,dateStr)

series     = '' # str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = '' # os.path.join(sweepPath,series)

# ## Create a driver instance
# driver = laser_driver.LaserDriver()

def get_paths():
    ## Sub directory definitions
    dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
    sweepPath = os.path.join(dataPath,dateStr)

    series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    seriesPath = os.path.join(sweepPath,series)

    return dateStr, sweepPath, series, seriesPath

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Acquire a laser timestream with the USRP using the GPU_SDR backend.')

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
    parser.add_argument('--timeLaser', '-Tl' , type=float, default=duration, 
        help='Duration of the laser/LED scan in seconds (default '+str(duration)+' seconds)')
    parser.add_argument('--ledDec', '-dc' , type=float, default=led_dec, 
        help='Decimation factor for the laser/LED timestreams (default '+str(led_dec)+'x)')

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

        powers[0] = args.power
        n_pwrs = len(powers)
        print("All Powers:", powers, "(",n_pwrs,")")

        min_pwer = -70.0
        max_pwer = -15.0
        for i in np.arange(n_pwrs):
            if (powers[i] < min_pwer):
                print("Power",args.power,"too Low! Range is "+str(min_pwer)+" to "+str(max_pwer)+" dBm. Adjusting to minimum...")
                powers[i] = min_pwer

            # Don't need to enforce this because it is used to tune up the tx gain later
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

def temp_change_and_wait(new_sp_K,nf_inst):

    print("CHANGING SETPOINT TO",new_sp_K*1e3,"mK")
    nf_inst.setSP(new_sp_K)

    cTemp = None

    while cTemp is None:
        try:
            cTemp=float(nf_inst.getTemp())
        except:
            print("Socket Failed, trying again soon")
            sleep(sleepTime)

    print("Waiting for Fridge to Reach Temperature")
    print("Monitoring temp every",sleepTime,"seconds")
    print("...",cTemp*1e3,"mK")
    terr = new_sp_K-cTemp
    ## This part doesn't work because getTemp only queries the setpoint
    while(np.abs(terr) > tempTolerance):
        sleep(sleepTime)
        try:
            cTemp=float(nf_inst.getTemp())
            terr = new_sp_K-cTemp
            print("...",cTemp*1e3,"("+str(terr*1e3)+") mK")
        except:
            print("Socket Failed, skipping reading")

    print("Holding at current temp for",stableTime,"seconds")
    sleep(stableTime)

    print("Done.")
    return 0

def runLaser(tx_gain, rx_gain, _iter, rate, freq, front_end, fspan, lapse_VNA, lapse_noise, lapse_laser, laser_dec, points, ntones, delay_duration, delay_over=None, h5_group_obj=None):
    
    delay = puif.run_delay(series=series, 
        tx_gain      = tx_gain, 
        rx_gain      = rx_gain, 
        rate         = rate, 
        freq         = freq, 
        front_end    = front_end, 
        lapse_delay  = delay_duration, 
        delay_over   = delay_over, 
        h5_group_obj = h5_group_obj)

    f, q = puif.run_vna(series=series, 
        res          = res, 
        tx_gain      = tx_gain, 
        rx_gain      = rx_gain,  
        _iter        = _iter, 
        rate         = rate, 
        freq         = freq, 
        front_end    = front_end,  
        fspan        = fspan, 
        lapse_VNA    = lapse_VNA, 
        points       = points, 
        ntones       = ntones, 
        h5_group_obj = h5_group_obj)

    cal_freqs, cal_means = puif.run_noise(series=series,
        delay        = delay, 
        f            = f, 
        q            = q, 
        cal_deltas   = cal_deltas, 
        tracking_tones = tracking_tones, 
        tx_gain      = tx_gain, 
        rx_gain      = rx_gain,  
        rate         = rate, 
        freq         = freq, 
        front_end    = front_end,  
        lapse_noise  = lapse_noise,
        cal_lapse_sec = cal_lapse_sec, 
        points       = points, 
        ntones       = ntones, 
        h5_group_obj = h5_group_obj,
        idx          = None)

    ## We'll need this for when we do multiple noise characterization scans
    cal_fs_obj = np.zeros(shape=(2,len(cal_freqs))                   ) ; cal_fs_obj[1] = cal_freqs
    cal_ms_obj = np.zeros(shape=(2,len(cal_means)),dtype='complex128') ; cal_ms_obj[1] = cal_means

    ## Now take a laser run, with no calibration deltas
    readout_tones  = np.append([f], tracking_tones)
    n_ro_tones     = len(readout_tones)
    readout_tones  = np.around(readout_tones, decimals=0)

    ## Split the power evenly across the tones
    amplitudes     = 1./ntones * np.ones(n_ro_tones)

    relative_tones = np.zeros(n_ro_tones)
    for k in np.arange(n_ro_tones):
        relative_tones[k] = float(readout_tones[k]) - freq

    ## Connect to the DC supply and confiure/enable the output
    e3631a.focusInstrument()
    print("*IDN?", e3631a.getIdentity())
    e3631a.clearErrors()
    # e3631a.doSoftReset()

    ## Declare a flag for additional noise characterization scans
    adtl_noise_scan = False
    
    ## Loop until the user opts to quit
    for iv in np.arange(len(LED_voltages)):
        V_led = LED_voltages[iv]

        ## Show the user the voltage then update the output file
        print("Using an LED voltage of:","{:.3f}".format(V_led),"V")
        outfname = "USRP_LaserOn_"+"{:.3f}".format(V_led)+"V_"+series

        ## Set the DC power supply output voltage
        e3631a.focusInstrument()
        e3631a.setOutputState(enable=True)
        e3631a.setVoltage(V_led)

        ## Turn on the AWG output
        fg3102.focusInstrument()
        fg3102.setOutputState(enable=True)

        ## Create a group for the noise scan parameters
        gScan = h5_group_obj.create_group('LaserScan_'+"{:.3f}".format(V_led).replace(".","-")+'V')
        gScan.attrs.create("file",  outfname+".h5")
        gScan.create_dataset("readout_tones",  data=readout_tones)
        gScan.create_dataset("relative_tones", data=relative_tones)
        gScan.create_dataset("amplitudes",     data=amplitudes)
        gScan.create_dataset("LOfrequency",    data=np.array([freq]))
        gScan.create_dataset("LEDvoltage",     data=np.array([V_led]))
        gScan.create_dataset("LEDfreqHz",      data=np.array([afg_pulse_params["f_Hz"]]))
        gScan.create_dataset("LEDpulseus",     data=np.array([afg_pulse_params["pw_us"]]))
        gScan.create_dataset("LEDVhi",         data=np.array([afg_pulse_params["V_hi"]]))
        gScan.create_dataset("LEDVlo",         data=np.array([afg_pulse_params["V_lo"]]))
        gScan.create_dataset("delayms",        data=np.array([afg_pulse_params["d_ms"]]))

        ## Determine how long to acquire noise
        # dur_laser = lapse_laser if ((np.abs(delta) < 0.005) and (cal_lapse_sec < lapse_noise)) else cal_lapse_sec  ## passed in sec
        gScan.create_dataset("duration",       data=np.array([lapse_laser]))
        
        print("Starting Laser/LED Run...")
        ## Do a noise run with the USRP
        laser_file = u.get_tones_noise(relative_tones, 
                                    measure_t  = lapse_laser, 
                                    tx_gain    = tx_gain, 
                                    rx_gain    = rx_gain, 
                                    rate       = rate,  ## passed in Hz
                                    decimation = laser_dec, 
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
        laser_file += '.h5'

        ## Wait for the chip to cool off?
        print("Waiting for chip to cool...")
        time.sleep(sleep_sec) ## 30 seconds

        ## Turn off the AWG output
        fg3102.focusInstrument()
        fg3102.setOutputState(enable=False)

        ## Stop putting out an LED voltage
        e3631a.focusInstrument()
        e3631a.setVoltage(0.0)
        # e3631a.setOutputState(enable=False)

        ## After three LED blasts, do a new noise run
        if ( (iv+1) % 3 == 0):
            adtl_noise_scan = True

            cal_freqs2, cal_means2 = puif.run_noise(series=series,
                delay        = delay, 
                f            = f, 
                q            = q, 
                cal_deltas   = cal_deltas, 
                tracking_tones = tracking_tones, 
                tx_gain      = tx_gain, 
                rx_gain      = rx_gain,  
                rate         = rate, 
                freq         = freq, 
                front_end    = front_end,  
                lapse_noise  = lapse_noise,
                cal_lapse_sec = cal_lapse_sec, 
                points       = points, 
                ntones       = ntones, 
                h5_group_obj = h5_group_obj,
                idx          = iv)

            cal_fs_obj = np.r_[cal_fs_obj, [cal_freqs2]]
            cal_ms_obj = np.r_[cal_ms_obj, [cal_means2]]

    if adtl_noise_scan:
        return cal_fs_obj[1:,:], cal_ms_obj[1:,:]
    else:
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

    cal_freqs, cal_means = runLaser(
        tx_gain = args.txgain,
        rx_gain = args.rxgain,
        _iter   = args.iter,
        rate    = args.rate,        ## Passed in Samps/sec
        freq    = args.LOfrq,       ## Passed in Hz
        front_end = "A",
        fspan   = args.VNAfspan,      ## Passed in Hz
        # f0      = args.f0,          ## Passed in Hz, relative to LO
        # f1      = args.f1,          ## Passed in Hz, relative to LO
        lapse_VNA   = args.timeVNA,   ## Passed in seconds
        lapse_noise = args.timeNoise, ## Passed in seconds
        lapse_laser = args.timeLaser, ## Passed in seconds
        laser_dec   = args.ledDec,
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

## This is intended to take sporadic LED data as the fridge cools down.
## To do so it reads the temperature  at the MC every so often and when 
## it   passes  user-defined   thresholds   it   starts  taking   data. 
## Alternatively,  it could be made  such that it takes LED at constant 
## intervals, like every hour, once the temperture has crossed a single 
## threshold
if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Connect to GPU SDR server
    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit(1)

    ## Instantiate the GPIB devices
    e3631a = E3631A()
    fg3102 = AFG3102()

    ## Initialize the NEXUS MGC3 servers
    nf2 = NEXUSThermometer()
    nf3 = NEXUSHeater()#server_ip="192.168.0.34",server_port=11034)

    ## Configure the AWG output
    fg3102.configureGPIB()
    fg3102.focusInstrument()
    print("*IDN?", fg3102.getIdentity())
    fg3102.clearErrors()
    fg3102.doSoftReset()

    fg3102.configureSource(afg_pulse_params)
    fg3102.setOutputState(enable=False)

    ## Make sure the DC PS is not at voltage
    e3631a.focusInstrument()
    print("*IDN?", e3631a.getIdentity())
    e3631a.clearErrors()
    # e3631a.doSoftReset()
    e3631a.setVoltage(0.0)
    # e3631a.setOutputState(enable=False)

    ## Print some diagnostic text
    SP = float(nf3.getSP())
    print("Starting Set Point:",SP)
    print("Scan Settings")
    print("         Start Temp:",Temps[ 0]*1e3,"mK")
    print("           End Temp:",Temps[-1]*1e3,"mK")
    print("          Temp Step:",Temp_step*1e3,"mK")
    print("     Temp Tolerance:",tempTolerance*1e3,"mK")
    print("          Hold Time:",stableTime,"s")
    print("   Reading Interval:",sleepTime,"s")

    ## Run the temperature scan
    for T in Temps:
        
        ## Change the fridge temperature
        temp_change_and_wait(T, nf3)

        ## Loop over the powers considered
        for i in np.arange(n_pwrs):

            ## Pull the final fridge temperature
            start_T_mK   = None
            while start_T_mK is None:
                try:
                    start_T_mK   = nf2.getTemperature()*1e3
                except:
                    print("Socket Failed, trying again soon")
                    sleep(sleepTime)

            ## Create new file path for the data
            dateStr, sweepPath, series, seriesPath = get_paths()
            doRun(powers[i])

            ## Pull the final fridge temperature
            final_T_mK   = None
            while final_T_mK is None:
                try:
                    final_T_mK   = nf3.getTemperature()*1e3
                except:
                    print("Socket Failed, trying again soon")
                    sleep(sleepTime)

            ## Write some metadata to a file
            write_str    = series +","+ str(T) +","+ str(start_T_mK) +","+ str(end_T_mK) +"\n"
            with open(os.path.join(sweepPath,dateStr+"_run_temperatures.csv"), "a") as file:
                file.write(write_str)
    
    ## Disconnect from the USRP server
    u.Disconnect()



    

    