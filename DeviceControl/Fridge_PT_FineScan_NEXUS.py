import sys, os
import numpy as np
import datetime
import argparse

from time import sleep

## Point to the backend function scripts
# sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")
sys.path.insert(1, "../DeviceControl")
try:
    from NEXUSFunctions import * #control NEXUS fridge
except ImportError:
    print("Cannot find the NEXUSFunctions package")
    exit()

## Flag to determine direction of temperature scan
start_at_max_T = False
return_to_base = False

## Temperature scan settings [K]
Temp_base =  15.0e-3
Temp_min  =  15.0e-3
Temp_max  = 125.0e-3
Temp_step =   5.0e-3
substepK  =   1.0e-3

## Temperature stabilization params
tempTolerance =   1e-4      ## K
tempTolFrac   =   0.005     ## Fraction of SP to wait for stability, picked by max(this,absTempTol)
sleepTime     =  30.0       ## sec
stableTime    =  40.0 * 60. ## sec
stableTimeSub =   0.5 * 60. ## sec

## Create the temperature array
Temps = np.arange(Temp_min,45.0e-3+Temp_step,Temp_step)
Temp_step = 2.0*Temp_step
Temps = np.append(Temps , np.arange(45.0e-3+Temp_step,Temp_max,Temp_step))

print(Temps)

if start_at_max_T:
    ## Use this if starting at the top temperature
    Temps = Temps[::-1] 
    if (Temp_base) < Temps[-1]:
        Temps = np.append(Temps,Temp_base)
else:
    # Use this if starting at base temperature
    if (Temp_base) < Temps[0]:
       Temps = np.append(Temp_base,Temps)

def temp_change_and_wait(new_sp_K,nf_inst,waittime=stableTime):
    try:
        nf_inst.setSP(new_sp_K)
    except:
        print("Socket Failed, trying again soon")
        sleep(sleepTime)
        nf_inst.setSP(new_sp_K)

    cTemp = None

    while cTemp is None:
        try:
            cTemp=float(nf_inst.getTemp())
        except:
            print("Socket Failed, trying again soon")
            sleep(sleepTime)


    print("Waiting for Fridge to Reach Temperature")
    print("Monitoring temp every",sleepTime,"seconds")
    print("...",cTemp*1e3,"mK")
    terr = new_sp_K-cTemp

    tempTol = np.max([tempTolerance, tempTolFrac*new_sp_K])

    while(np.abs(terr) > tempTol):
        sleep(sleepTime)
        try:
            cTemp=float(nf_inst.getTemp())
            terr = new_sp_K-cTemp
            print("...",cTemp*1e3,"("+str(terr*1e3)+") mK")
        except:
            print("Socket Failed, skipping reading")

    print("Holding at current temp for",waittime,"seconds")
    sleep(waittime)

    print("Done.")
    return 0

if __name__ == "__main__":
    ## Initialize the NEXUS MGC3/MMR3 servers
    nf2 = NEXUSThermometer()
    nf3 = NEXUSHeater()

    ## Print some diagnostic text
    SP = float(nf3.getSP())
    print("Starting Set Point:",SP)
    print("Scan Settings")
    print("         Start Temp:",Temps[ 0]*1e3,"mK")
    print("           End Temp:",Temps[-1]*1e3,"mK")
    print("          Temp Step:",Temp_step*1e3,"mK")
    print("     Temp Tolerance:",tempTolerance*1e3,"mK")
    print("          Hold Time:",stableTime,"s")
    print("   Reading Interval:",sleepTime,"s")

    ## Run the temperature scan
    for T in Temps:
        
        ## Change the fridge temperature
        temp_change_and_wait(T, nf3)

        ## After the power scan, do a series of fine steps to allow others to take data
        ## First, stop if we're at the last temperature
        if T == Temps[-1]:
            break

        if start_at_max_T:
            ## Create an array of temperatures assuming you're going down
            subTemps = np.arange(start=T-Temp_step+substepK,stop=T,step=substepK)
            subTemps = subTemps[::-1]
        else:
            ## Create an array of temperatures assuming you're going up
            subTemps = np.arange(start=T+substepK,stop=T+Temp_step,step=substepK)

        for sT in subTemps:
            temp_change_and_wait(sT, nf3, waittime=stableTimeSub)

    ## Go back to base temperature
    if return_to_base:
        print("Reverting to base temperature of",Temp_base*1e3,"mK")
        nf3.setSP(Temp_base)


