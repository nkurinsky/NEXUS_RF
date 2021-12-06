from __future__ import division
import sys
import numpy as np
import time
sys.path.append('PyMKID')
import PyMKID_USRP_import_functions as puf2
import os
#import matplotlib.pyplot as plt
#import h5py
sys.path.append('GPU_SDR')
import pyUSRP as u

if not u.Connect():
    u.print_error("Cannot find the GPU server!")
    exit()

#def get_raw(openfile):
#    raw_data = np.array(openfile["raw_data0/A_RX2/data"])
#    return np.transpose(raw_data)

#def vna_run(gain,iter,rate,freq,front_end,f0,f1,lapse,points,ntones,delay_duration,delay_over):
#
#    if delay_over is not None:
#        delay = 1e-6*delay_over
#
#    try:
#        if u.LINE_DELAY[str(int(rate/1e6))]: pass
#    except KeyError:
#
#        if delay_over is None:
#            print "Cannot find line delay. Measuring line delay before VNA:"
#
#            filename = u.measure_line_delay(rate, freq, front_end, USRP_num=0, tx_gain=0, rx_gain=0, output_filename=None, compensate = True, duration = delay_duration)
#
#            delay = u.analyze_line_delay(filename, True)
#
#            u.write_delay_to_file(filename, delay)
#
#            u.load_delay_from_file(filename)
#
#        else:
#
#            u.set_line_delay(rate, delay_over)
#
#        if ntones ==1:
#            ntones = None
#
#    vna_filename = u.Single_VNA(start_f = f0, last_f = f1, measure_t = lapse, n_points = points, tx_gain = gain, Rate=rate, decimation=True, RF=freq, Front_end=front_end,
#               Device=None, output_filename=None, Multitone_compensation=ntones, Iterations=iter, verbose=False)
#
#    return vna_filename, delay*1e9

if __name__ == '__main__':

    #try:
    #    import pyUSRP as u
    #except ImportError:
    #    try:
    #        sys.path.append('/home/dmkidaq/GPU_SDR')
    #        import pyUSRP as u
    #    except ImportError:
    #        print "Cannot find the pyUSRP package"

    #folder = '/home/dmkidaq/usrp_data/200804'

    #try:
    #    os.mkdir(folder)
    #except OSError:
    #    pass

    #os.chdir(folder)

    #if not u.Connect():
    #    u.print_error("Cannot find the GPU server!")
    #    exit()

    f0 = 0e6
    f1 = 90e6
    LO = 4.2e9
    run = True
    duration = 4
    rx_gain = 17.5
    tx_gain = 0
    rate = 200e6


    power = -40
    N_power = 10**(((-1*power)-14)/20)
    print round(-14-20*np.log10(N_power),2),' dBm'
    print N_power

    delay_filename = u.measure_line_delay(rate, LO, 'A', USRP_num=0, tx_gain=tx_gain, rx_gain=rx_gain, output_filename=None, compensate = True, duration = 0.01)
    delay = u.analyze_line_delay(delay_filename, True)
    print(delay)
    os.remove(delay_filename+'.h5')
    while run == True:
        try:
            #file0, delay0 = vna_run(gain=0,iter=1,rate=200e6,freq=LO,front_end='A',f0=f0,f1=f1,lapse=1,points=450000,ntones=3+0.5*ntones_exp,delay_duration=0.01,delay_over=0)
            #file0, delay0 = vna_run(gain=ntones_exp,iter=1,rate=200e6,freq=LO,front_end='A',f0=f0,f1=f1,lapse=1,points=450000,ntones=1,delay_duration=0.01,delay_over=0)

            file0, _ = puf2.vna_run(tx_gain=tx_gain,rx_gain=rx_gain,iter=1,rate=200e6,freq=LO,front_end='A',f0=f0,f1=f1,lapse=duration,points=450000,ntones=N_power,delay_duration=0.01,delay_over=delay)
            #file0, delay0 = vna_run(gain=0,iter=1,rate=200e6,freq=LO,front_end='A',f0=f0,f1=f1,lapse=1,points=450000,ntones=10**(15+ntones_exp),delay_duration=0.01,delay_over=0)
            np.savetxt(file0+'.delete',[0])
            #ntones_exp += 1
            #ntones_exp = ntones_exp%5
            print('pausing for two seconds for when you want to quit; ctrl+c to quit')
            time.sleep(2)

        except KeyboardInterrupt:
            run = False
            print 'This is the except region'

    print 'Done'

    #with h5py.File(file0+'.h5', "r") as fyle:
    #    raw_VNA = get_raw(fyle)
    #    #amplitude = fyle["raw_data0/A_RX2"].attrs.get('ampl')
    #    rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
    #    LO = fyle["raw_data0/A_RX2"].attrs.get('rf')

    #raw_f = np.arange(LO+f0, LO+f1, (f1-f0)/len(raw_VNA[:,0]))
    #plt.plot(raw_f,20*np.log10(abs(raw_VNA)))
    #plt.show()
