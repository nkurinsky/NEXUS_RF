import sys
sys.path.append("../Devices/")

import numpy
from ADRfunctions import *
from time import sleep

timeInterval=10
delayTime_min=5 #minutes
minTemp=85
maxTemp=240
tStep=5

delayTime=delayTime_min*60
nSteps=int(delayTime/timeInterval)

fridge=ADR()

scaleFactor=1.0/1.25*1e-3
temps = numpy.arange(minTemp,maxTemp,tStep)
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

print("Done!")
fridge.rampOff()
