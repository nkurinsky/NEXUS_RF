#CMT VNA script
from __future__ import division
import sys, os
import numpy as np
import datetime
from time import sleep
sys.path.append("../DeviceControl/")

datapath='/data/TempSweeps/VNA'
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

def getTempStr(temp):
    if(temp > 0.099):
        return str(temp*1e3)[:3]
    else:
        return str(temp*1e3)[:2]

timeInterval=10
delayTime_min=5 #minutes
minTemp=85
maxTemp=250
tStep=5
n_avs = 10
#wait_secs = 65
#if_bandwidth = 10e3

debugPowerScan=False

powers = [-50,-47.5,-45,-42.5,-40,-37.5,-35,-32.5,-30]

delayTime=delayTime_min*60
nSteps=int(delayTime/timeInterval)

fridge=ADR()
v = VNA()

temps = np.arange(minTemp,maxTemp+tStep,tStep)
nTemps = len(temps)

print("Temperature Scan")
print("Min:",minTemp,"Max:",maxTemp,"TStep:",tStep)
print("Time per Step:",delayTime_min,"minutes")

for i in range(0,nTemps):
    temp=temps[nTemps-1-i]*1e-3
    print("#Target Temp:",temp)

    cTemp=fridge.getTemp()
    print(cTemp,temp)
    if(abs(cTemp - temp) > 1e-3):
        if(debugPowerScan == False):
            fridge.setTempSP(temp)
        for i in range(0,nSteps):
            sleep(timeInterval)
            cTemp=fridge.getTemp()
            print("#  ",i,cTemp)
    else:
        print("No Ramp Needed")

    cTempStr=getTempStr(cTemp)
    
    print("Temp:"+cTempStr+", Starting VNA Scan")
    
    for power in powers:
        print(str(round(power)) + ' dBm of power')
        
        output_filename = sweepPath+"/TPsweep"+"_T"+cTempStr+"_P"+str(power)+"_"+series
        
        v.setPower(power)
        freqs, S21_real, S21_imag = v.takeSweep(4241e6, 4246e6, 5e4, n_avs)

        v.storeData(freqs, S21_real, S21_imag, output_filename)

print("Done!")
fridge.rampOff()
