from __future__ import division
import sys,os
import numpy as np
try:
    import pyUSRP as u
except ImportError:
    try:
        sys.path.append('GPU_SDR')
        import pyUSRP as u
    except ImportError:
        print("Cannot find the pyUSRP package")

sys.path.append('PyMKID')
import PyMKID_USRP_functions as puf
import PyMKID_USRP_import_functions as puf2
if not u.Connect():
    u.print_error("Cannot find the GPU server!")
    exit()



powers = [-50, -45, -40]
for power in powers:

	N_power = 10**(((-1*power)-14)/20)

	print(str(round(-14-20*np.log10(N_power),2)) + ' dBm of power')
	print (str(N_power) + ' is the equivalent number of tones needed to split the DAQ power into the above amounut')



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
		                       delay_over='null')

u.Disconnect()
