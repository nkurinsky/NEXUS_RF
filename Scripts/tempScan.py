import sys, os
from __future__ import division
import numpy as np
sys.path.append("../Devices/")

from ADRfunctions import *
from time import sleep

try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('~/workarea/GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

sys.path.append('~/workarea/PyMKID')
import PyMKID_USRP_functions as puf
import PyMKID_USRP_import_functions as puf2
if not u.Connect():
    u.print_error("Cannot find the GPU server!")
    exit()


timeInterval=10
delayTime_min=5 #minutes
minTemp=85
maxTemp=240
tStep=5

powers = [-70,-65,-60,-55,-50,-45,-40,-35,-30]

delayTime=delayTime_min*60
nSteps=int(delayTime/timeInterval)

fridge=ADR()

scaleFactor=1.0/1.25*1e-3
temps = np.arange(minTemp,maxTemp,tStep)
nTemps = len(temps)

print("Temperature Scan")
print("Min:",minTemp,"Max:",maxTemp,"TStep:",tStep)
print("Time per Step:",delayTime_min,"minutes")

for i in range(0,nTemps):
    temp=temps[nTemps-1-i]
    setPoint = temp*scaleFactor
    print("#Target Temp:",temp,"Set Point:",setPoint)

    fridge.setSP(setPoint)
    for i in range(0,nSteps):
        sleep(timeInterval)
        cTemp=fridge.getTemp()
        print("#  ",i,cTemp)

    print(setPoint*1e3,cTemp*1e3)
    cTemp=fridge.getTemp()
    #doVNA Scan
    for power in powers:
        N_power = 10**(((-1*power)-14)/20)
        print(str(round(-14-20*np.log10(N_power),2)) + ' dBm of power')
        print (str(N_power) + ' is the equivalent number of tones needed to split the DAQ power into the above amount')
        
        output_filename = "TPsweep"+"_T"+str(cTemp)+"_P"+str(power)+"__"+str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))

        vna_file, delay = puf2.vna_run(tx_gain=0, \
                                       rx_gain = 25,\
                                       iter=1,\
                                       rate=200e6,
                                       freq=4.2e9,\
                                       front_end='A',\
                                       f0=1e6,f1=6e6,\
                                       lapse=20,\
                                       points=1e5,\
                                       ntones=N_power,\
                                       delay_duration=0.01,\
                                       delay_over='null',\
                                       output_filename=output_filename)

print("Done!")
fridge.rampOff()
u.Disconnect()
