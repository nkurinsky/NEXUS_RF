from __future__ import division
import sys,os
import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('../../Devices/GPU_SDR_old')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

# sys.path.append('PyMKID')
import PyMKID_USRP_functions as puf
import PyMKID_USRP_import_functions as puf2

power = -50
freq = 4.24e9
f0 = 1e6
f1 = 4e6
points = 1e3
tx_gain=0
rx_gain=0
filename=None

narg = len(sys.argv)
if(narg < 4):
    print("Calling Sequence: python2 take_VNA.py l0 f0 f1 [tx power] [filename]")
    print("Using stored defaults")
else:
    freq = float(sys.argv[1])
    f0 = float(sys.argv[2])
    f1 = float(sys.argv[3])
    if((f1 - f0) > 1e7):
        print("Frequency range too large")
        exit(1)
    if(freq>6e9):
        print("Invalid LO Frequency")
if(narg > 4):
    power=float(sys.argv[4])
    if(power < -70):
        print("Power too Low! Range is -70 to 0")
        exit(2)
    elif(power > 0):
        print("Power too High! Range is -70 to 0")
        exit(2)
if(narg > 5):
    filename=sys.argv[5]

if not u.Connect():
    u.print_error("Cannot find the GPU server!")
    exit()

print('VNA Settings')
print('LO: '+str(freq))
print('f0: '+str(f0))
print('f1: '+str(f1))
print('power: '+str(power))
print('tx gain: '+str(tx_gain))
print('rx gain: '+str(rx_gain))
print('npoints: '+str(points))
if(filename != None):
    print('Filename: '+filename)
else:
    print('Using default filename')

N_power = 10**(((-1*power)-14)/20)

print(str(round(-14-20*np.log10(N_power),2)) + ' dBm of power')
print (str(N_power) + ' is the equivalent number of tones needed to split the DAQ power into the above amounut')

vna_file, delay = puf2.vna_run(tx_gain=tx_gain, \
		               rx_gain = rx_gain,\
		               iter=1,\
		               rate=200e6,
		               freq=freq,\
		               front_end='A',\
		               f0=f0,f1=f1,\
		               lapse=20,\
		               points=points,\
		               ntones=N_power,\
		               delay_duration=0.01,\
		               delay_over='null',\
                               output_filename=filename)

u.Disconnect()
