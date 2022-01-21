from __future__ import division

## Import the relevant modules
import sys, os
import numpy as np
import time
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

try:
    import PyMKID_USRP_import_functions as puf2
except ImportError:
    print("Cannot find the PyMKID_USRP_import_functions package")
    exit()

## Attempt to connect to GPU SDR server
if not u.Connect():
    u.print_error("Cannot find the GPU server! Exiting.")
    quit()

## Create some directories for the data files
dataPath = '/data/UsrpNoiseScans'
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

dateStr    = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = dataPath + '/' + series 
if not os.path.exists(seriesPath):
    os.makedirs(seriesPath)
print ("Scan stored as series "+series+" in path "+dataPath)

    
## Set some noise scan parameters
rate    = 200e6
tx_gain = 0
rx_gain = 17.5
LO      = 4.2e9
res     = 4.244905
tracking_tones = np.array([4.235e9,4.255e9])

## Set the stimulus powers to loop over
power = -35

## Ensure the power doesn't go above -25 dBm
if power > -25:
    USRP_power = -25
    tx_gain    = power - USRP_power
else:
    USRP_power = power

## Calculate some derived quantities
N_power = np.power(10.,(((-1*USRP_power)-14)/20.))
pwr_clc = np.round(-14-20*np.log10(N_power),2)

## Print some diagnostic text
print( pwr_clc,' dBm')
print( N_power)

## Do a VNA run with the USRP
vna_file, delay = puf2.vna_run(tx_gain        = tx_gain , 
                               rx_gain        = rx_gain ,
                               iter           = 1       ,
                               rate           = rate    ,
                               freq           = LO      ,
                               front_end      = 'A'     ,
                               f0             = 43e6    ,
                               f1             = 48e6    ,
                               lapse          = 20      ,
                               points         = 1e5     ,
                               ntones         = N_power ,
                               delay_duration = 0.1     ,
                               delay_over     = 'null'  ,
                               subfolder      = seriesPath )

## Fit the data acquired in this noise scan
fs, qs, _,_,_,_,_ = puf.vna_file_fit(vna_file + '.h5',[res],show=False)

## Extract the important parameters from fit
f = fs[0]*1e9
q = qs[0]

## Disconnect from the USRP server
u.Disconnect()
