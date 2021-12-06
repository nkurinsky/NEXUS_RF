#USRP Script
from __future__ import division
import sys, os
import numpy as np
import datetime
from time import sleep
sys.path.append("../Devices/")

datapath='/data/TempSweeps'
if not os.path.exists(datapath):
    os.makedirs(datapath)

dateStr=str(datetime.datetime.now().strftime('%Y%m%d'))
sweepPath=datapath+'/'+dateStr
if not os.path.exists(sweepPath):
    os.makedirs(sweepPath)

series=str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print ("Scan stored as series "+series+" in path "+sweepPath)

from ADRfunctionsUSRP import *

def getTempStr(temp):
    if(temp > 0.099):
        return str(temp*1e3)[:3]
    else:
        return str(temp*1e3)[:2]

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('/home/nexus-admin/workarea/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

sys.path.append('/home/nexus-admin/workarea/PyMKID')
import PyMKID_USRP_functions as puf
import PyMKID_USRP_import_functions as puf2

timeInterval=10
delayTime_min=5 #minutes
minTemp=85
maxTemp=250
tStep=5

debugPowerScan=False

powers = [-70,-65,-60,-55,-50,-45,-40,-35,-30]

delayTime=delayTime_min*60
nSteps=int(delayTime/timeInterval)

fridge=ADR()

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

    if not u.Connect():
        u.print_error("Cannot find the GPU server!")
        exit()
    
    for power in powers:
        N_power = 10**(((-1*power)-14)/20)
        print(str(round(-14-20*np.log10(N_power),2)) + ' dBm of power')
        print (str(N_power) + ' is the equivalent number of tones needed to split the DAQ power into the above amount')
        
        output_filename = sweepPath+"/TPsweep"+"_T"+cTempStr+"_P"+str(power)+"_"+series
        
        vna_file, delay = puf2.vna_run(tx_gain=0, \
                                       rx_gain = 25,\
                                       iter=1,\
                                       rate=200e6,
                                       freq=4.24e9,\
                                       front_end='A',\
                                       f0=1e6,f1=6e6,\
                                       lapse=5,\
                                       points=5e4,\
                                       ntones=N_power,\
                                       delay_duration=0.01,\
                                       delay_over='null',\
                                       output_filename=output_filename)

    u.Disconnect()

print("Done!")
fridge.rampOff()
