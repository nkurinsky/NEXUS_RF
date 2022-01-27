import sys, os
from time import sleep
import numpy as np
import datetime

## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")

from VNAfunctions import *  #using the VNA to do a power sweep
from NEXUSFunctions import * #control NEXUS fridge
from VNAMeas import * #vna measurement class

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

# Initialize the NEXUS temperature servers
nf1 = NEXUSTemps(server_ip="192.168.0.31",server_port=11031)
nf2 = NEXUSTemps(server_ip="192.168.0.32",server_port=11032)

## Parameters of the power sweep (in dB)
P_min  = -45.0
P_max  = -20.0
P_step =   5.0

powers = np.arange(start = P_min,
                   stop  = P_max+P_step,
                   step  = P_step)
print("Scanning over powers (dB):", powers)

## How many readings to take at each step of the sweep
n_avs = 10

# Diagnostic text
print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())

print("Power Scan Settings")
print("   Start Power (dB):", P_min)
print("     End Power (dB):", P_max)
print("    Power Step (dB):", P_step)
print(" N Points Avgeraged:", n_avs)

## Initialize the VNA
v = VNA()

## Set the VNA's frequency parameters
freqmin = 4.24205e9 # 4.24212e9   ## Hz
freqmax = 4.24225e9 # 4.24262e9   ## Hz
n_samps = 5e4

## Start the power loop
for power in powers:
  print("Current Power (dBm):", str(round(power)))

  ## Create a filename for this sweep
  output_filename = seriesPath +"/Psweep_P"+str(power)+"_"+series

  ## Create a class to contain the sweep result
  sweep = VNAMeas(dateStr, series)
  sweep.power   = power
  sweep.n_avgs  = n_avs
  sweep.n_samps = n_samps
  sweep.f_min   = freqmin
  sweep.f_max   = freqmax

  ## Grab and save the fridge temperature before starting sweep
  sweep.start_T = np.array([nf1.getTemp(), nf2.getTemp()])

  ## Set the VNA stimulus power and take a frequency sweep
  v.setPower(power)
  freqs, S21_real, S21_imag = v.takeSweep(freqmin, freqmax, n_samps, n_avs)

  ## Grab and save the fridge temperature after sweep
  sweep.final_T = np.array([nf1.getTemp(), nf2.getTemp()])

  ## Save the result to our class instance
  sweep.frequencies = np.array(freqs)
  sweep.S21realvals = np.array(S21_real)
  sweep.S21imagvals = np.array(S21_imag)

  ## Write our class to a file
  print("Storing data at:", sweep.save_hdf5(output_filename))

  ## Store the data in our file name
  v.storeData(freqs, S21_real, S21_imag, output_filename)

## Diagnostic text
print("Current Fridge Temperature 1 (mK): ", nf1.getTemp())
print("Current Fridge Temperature 2 (mK): ", nf2.getTemp())
print("Power scan complete.")
