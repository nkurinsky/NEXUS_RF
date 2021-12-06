import sys
sys.path.append("../Devices")
from time import sleep
import numpy as np

from NEXUSFunctions import *

nf = NEXUSFridge()

tStart = 20e-3
tEnd = 60e-3
dt = 10e-3

tempTolerance=5e-4
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


print("Reverting to Initial Setpoint")
nf.setSP(oSP)
