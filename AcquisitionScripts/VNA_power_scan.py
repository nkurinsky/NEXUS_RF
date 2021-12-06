#CMT VNA script
from __future__ import division
import sys, os
import numpy as np
import datetime
from time import sleep
sys.path.append("../Devices/")

datapath='/data/PowerSweeps/VNA'
if not os.path.exists(datapath):
    os.makedirs(datapath)

dateStr=str(datetime.datetime.now().strftime('%Y%m%d'))
sweepPath=datapath+'/'+dateStr
if not os.path.exists(sweepPath):
    os.makedirs(sweepPath)

series=str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print ("Scan stored as series "+series+" in path "+sweepPath)

from ADRfunctionsUSRP import *
from VNAfunctions import *

n_avs = 30 
#wait_secs = 65
#if_bandwidth = 10e3

debugPowerScan=False

powers = [-55,-50,-45,-40,-35,-30,-25,-20]

fridge=ADR()
v = VNA()

for power in powers:
    print(str(round(power)) + ' dBm of power')
     
    output_filename = sweepPath+"/Psweep"+"_P"+str(power)+"_"+series
        
    v.setPower(power)
    freqs, S21_real, S21_imag = v.takeSweep(4.24212e9, 4.24262e9, 5e4, n_avs)

    v.storeData(freqs, S21_real, S21_imag, output_filename)

print("Done!")

