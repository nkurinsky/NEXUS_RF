## Import the relevant modules
import sys, os
import time, datetime
import h5py
import numpy as np

## Import device control drivers
try:
    from E3631A  import *
    from AFG3102 import *
except ImportError:
    try:
        sys.path.append('../DeviceControl')
        from E3631A  import *
        from AFG3102 import *
    except ImportError:
        print("Cannot find the GPIB device drivers package")
        exit()

## Import the usb spectrometer driver
#try: 
#    import stellarnet_driver as sn
#except ImportError:
#    print("Cannot find the StellarNet Spectrometer driver")
#    exit()

try:
    from stellarnet_driverLibs import stellarnet_driver3 as sn
except:
    print("\n\n************************************ ERROR *****************************************")
    print("             ERROR:    Compatible Python Driver DOES NOT EXIST")
    print(" ** See \"stellarnet_driverLibs\" and documentation for all available compiled Drivers")
    print(" **      Contact Stellarnet Inc. ContactUs@StellarNet.us for additional support.")
    print("************************************ ERROR *****************************************\n\n")
    quit()

## Set Laser parameters
afg_pulse_params = {
    "f_Hz" :   1.0,
    "pw_us":  10.0,
    "V_hi" :   5.0,
    "V_lo" :   0.0,
    "d_ms" :  10.0,
    "NpB"  :   1.0,
}
LED_voltages = [3.0] # np.arange(start=2.000, stop=6.250, step=0.250)
N_pulses     = 100

## Set Spectrometer parameters
spectro_params = {
    "inttime": 10,
    "scansavg": 1,
    "smooth":   0,
    "xtiming":  1,
}

## File handling options
filename=None

## Where to save the output data (hdf5 files)
dataPath = '.'#'/data/Spectrometer_Data'

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

def create_dirs():
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    if not os.path.exists(sweepPath):
        os.makedirs(sweepPath)

    if not os.path.exists(seriesPath):
        os.makedirs(seriesPath)
    print ("Scan stored as series "+series+" in path "+sweepPath)
    return 0

def configure_awg(awg_obj, pulse_pars):
    ## Focus the instrument, reset it
    awg_obj.focusInstrument()
    print("*IDN?", awg_obj.getIdentity())
    awg_obj.clearErrors()
    awg_obj.doSoftReset()

    ## Make sure the AWG output is off
    awg_obj.setOutputState(enable=False)

    ## Configure the pulse parameters
    awg_obj.configureSource(pulse_pars)

    ## Reset parameters we care about
    awg_obj.updateNperCycle(int(pulse_pars["NpB"])) ## Only put out one LED pulse per trigger
    awg_obj._sendCmd("TRIGger:SOURce TIMer", getResponse=False) ## Use internal clock as trigger
    awg_obj._sendCmd("TRIGger:TIMer "+str(1./pulse_pars["f_Hz"]), getResponse=False) ## Fire once per second

# function to set device parameter and return spectrum
def getSpectrum(spectrometer, wav, spec_params):
    sn.setParam(spectrometer, spec_params["inttime"], spec_params["scansavg"], spec_params["smooth"], spec_params["xtiming"], True) 
    spectrum = sn.array_spectrum(spectrometer, wav)
    return spectrum


if __name__ == "__main__":

    ## Define the series for this batch of data
    dateStr, sweepPath, series, seriesPath = get_paths()

    ## Create the output directories
    create_dirs()

    ## Connect to the spectrometer instance
    spectrometer, wav = sn.array_get_spec(0)
    version = sn.version()
    print(version)
    print(spectrometer)
    sn.ext_trig(spectrometer, True)	

    ## Instantiate the GPIB devices
    e3631a = E3631A()
    fg3102 = AFG3102()

    ## Configure the GPIB-LAN interface output
    fg3102.configureGPIB()

    ## Set up the AWG
    configure_awg(fg3102,afg_pulse_params)

    ## Connect to and configure the DCPS
    e3631a.focusInstrument()
    print("*IDN?", e3631a.getIdentity())
    e3631a.clearErrors()
    e3631a.setVoltage(0.0)

    ## Now loop over every LED voltage we care about
    for iv in np.arange(len(LED_voltages)):
        V_led = LED_voltages[iv]

        ## Show the user the voltage then update the output file name
        print("Using an LED voltage of:","{:.3f}".format(V_led),"V")
        outfname = "Spectra_"+"{:.3f}".format(V_led)+"V_"+series+".h5"

        ## Instantiate an output file
        fyle = h5py.File(os.path.join(seriesPath,outfname),'w')

        ## Save some metadata to the h5 file
        fyle.attrs.create("LEDvoltage", V_led)
        fyle.attrs.create("LEDfreqHz",  afg_pulse_params["f_Hz"])
        fyle.attrs.create("LEDpulseus", afg_pulse_params["pw_us"])
        fyle.attrs.create("LEDVhi",     afg_pulse_params["V_hi"])
        fyle.attrs.create("LEDVlo",     afg_pulse_params["V_lo"])
        fyle.attrs.create("delayms",    afg_pulse_params["d_ms"])
        fyle.attrs.create("NumPerBurst",afg_pulse_params["NpB"])
        fyle.attrs.create("Npulses",    N_pulses)
        fyle.attrs.create("SpecIntTime",spectro_params["inttime"])
        fyle.attrs.create("SpecScanAvg",spectro_params["scansavg"])
        fyle.attrs.create("SpecSmooth", spectro_params["smooth"])
        fyle.attrs.create("SpecXTiming",spectro_params["xtiming"])

        ## Set the DC power supply output voltage
        e3631a.focusInstrument()
        e3631a.setOutputState(enable=True)
        e3631a.setVoltage(V_led)

        ## Now loop over how many LED pulses
        for i in np.arange(N_pulses):

            # ## Create an h5 group for this data, store some general metadata
            # gPulse = fyle.create_group('Pulse'+str(i))

            ## Turn on the AWG output, force a trigger
            fg3102.focusInstrument()
            fg3102.setOutputState(enable=True)
            fg3102._sendCmd("TRIG", getResponse=False)

            ## Take a spectrum from the spectrometer
            spectrum = getSpectrum(spectrometer, wav, spectro_params)

            ## Save the specturm to the data file
            fyle.create_dataset('Pulse'+str(i),data=spectrum)

            ## Turn off the AWG output
            fg3102.focusInstrument()
            fg3102.setOutputState(enable=False)

        ## Close h5 file for writing
        fyle.close()

    ## Stop putting out an LED voltage
    e3631a.focusInstrument()
    e3631a.setVoltage(0.0)