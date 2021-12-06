import sys, os
# sys.path.append("../Devices")
from time import sleep
import numpy as np
import datetime

## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")

from VNAfunctions import *  #using the VNA to do a power sweep
# from NEXUSFunctions import * #control NEXUS fridge

dataPath = '/data/PowerSweeps/VNA'  #VNA subfolder of TempSweeps
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

dateStr   = str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath = dataPath + '/' + dateStr
if not os.path.exists(sweepPath):
    os.makedirs(sweepPath)

series     = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
seriesPath = sweepPath + '/' + series 
if not os.path.exists(seriesPath):
    os.makedirs(seriesPath)
print ("Scan stored as series "+series+" in path "+sweepPath)


## Initialize the NEXUS fridge
# nf = NEXUSFridge()

#powers = [-55, -50, -45, -40, -35, -30, -25, -20]

## Parameters of the power sweep (in dB)
P_min  = -45.0
P_max  = -35.0
P_step =   1.0

powers = np.arange(start = P_min,
                   stop  = P_max+P_step,
                   step  = P_step)
print("Scanning over powers (dB):", powers)

## How many readings to take at each step of the sweep
n_avs = 10

## Parameters for each step of the sweep
sleepTime = 5
holdTime  = 60

## Diagnostic text
# print("Current Fridge Setpoint    (mK): ", nf.getSP())
# print("Current Fridge Temperature (mK): ", nf.getTemp())

print("Power Scan Settings")
print("   Start Power (dB):", P_min)
print("     End Power (dB):", P_max+P_step)
print("    Power Step (dB):", P_step)
print(" N Points Avgeraged:", n_avs)
print("          Hold Time:", holdTime , "sec")
print("   Reading Interval:", sleepTime, "sec")

## Initialize the VNA
v = VNA()

## Set the VNA's frequency parameters
freqmin = 4.24205e9 # 4.24212e9   ## Hz
freqmax = 4.24225e9 # 4.24262e9   ## Hz
n_samps = 5e4

## Start the power loop
for power in powers:
  print("Current Power (dBm):", str(round(power)))
  
  ## Create a filename for this acquisition
  output_filename = seriesPath +"/Psweep_P"+str(power)+"_"+series

  ## Set the VNA stimulus power and take a frequency sweep
  v.setPower(power)
  freqs, S21_real, S21_imag = v.takeSweep(freqmin, freqmax, n_samps, n_avs)
  # freqs, S21_real, S21_imag = v.takeSweep(4.24212e9, 4.24262e9, 5e4, n_avs)

  ## Store the data in our file name
  v.storeData(freqs, S21_real, S21_imag, output_filename)

## 
print("Power scan complete.")
