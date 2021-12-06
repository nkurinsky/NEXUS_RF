import sys, os
sys.path.append("../Devices")
from time import sleep
import numpy as np
import datetime

from VNAfunctions import *  #using the VNA to do a temperature sweep
from NEXUSFunctions import * #control NEXUS fridge

datapath='/data/TempSweeps/VNA'  #VNA subfolder of TempSweeps
if not os.path.exists(datapath):
    os.makedirs(datapath)

dateStr=str(datetime.datetime.now().strftime('%Y%m%d')) #sweep date
sweepPath=datapath+'/'+dateStr
if not os.path.exists(sweepPath):
    os.makedirs(sweepPath)

series=str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print ("Scan stored as series "+series+" in path "+sweepPath)


nf = NEXUSFridge()

powers = [-55, -50, -45, -40, -35, -30, -25, -20]

n_avs = 10

tStart = 20e-3
tEnd = 350e-3
dt = 10e-3

tempTolerance=1e-4
sleepTime=5
holdTime=60

temps = np.arange(tStart,tEnd+dt,dt)

oSP = float(nf.getSP())
print("Starting Set Point:",oSP)
print("Scan Settings")
print("         Start Temp:",tStart*1e3,"mK")
print("           End Temp:",tEnd*1e3,"mK")
print("          Temp Step:",dt*1e3,"mK")
print("     Temp Tolerance:",tempTolerance*1e3,"mK")
print("          Hold Time:",holdTime,"s")
print("   Reading Interval:",sleepTime,"s")

for temp in temps:
    print("Changing Setpoint to",temp,"K")
    nf.setSP(temp)

    cTemp=float(nf.getTemp())
    print("Waiting for Fridge to Reach Temperature")
    print("Monitoring temp every",sleepTime,"seconds")
    print("...",cTemp)
    terr = temp-cTemp
    while(np.abs(terr) > tempTolerance):
        sleep(sleepTime)
        try:
            cTemp=float(nf.getTemp())
            terr = temp-cTemp
            print("...",cTemp,"("+str(terr)+")")
        except:
            print("Socket Failed, skipping reading")

    print("Holding at current temp for",holdTime,"seconds")
    sleep(holdTime)

    v = VNA()

    for power in powers:
        print(str(round(power)) + ' dBm of power')

        powerPath = sweepPath + '/' + str(power)
        if not os.path.exists(powerPath):
            os.makedirs(powerPath)

        output_filename = powerPath +"/TPsweep"+"_T"+str(round(cTemp,5)*1000)+"_P"+str(power)+"_"+series

        v.setPower(power)
        freqs, S21_real, S21_imag = v.takeSweep(4.24212e9, 4.24262e9, 5e4, n_avs)

        v.storeData(freqs, S21_real, S21_imag, output_filename)


print("Reverting to base temperature")
nf.setSP(0.01)
