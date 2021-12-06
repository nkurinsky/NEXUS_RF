from __future__ import division
import matplotlib
import os
import numpy as np
import sys
sys.path.append('PyMKID')
import matplotlib.pyplot as plt
import PyMKID_USRP_functions as puf
from matplotlib import animation
import h5py
import time

#def get_raw(openfile):
#    raw_data = np.array(openfile["raw_data0/A_RX2/data"])
#    return np.transpose(raw_data)

if __name__ == '__main__':

    fig, ax = plt.subplots()
    plt.title('VNA')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('"S21" [dB]')
    last_raw_f = []
    last_data = []
    line0, = ax.plot(last_raw_f,last_data)
    line1, = ax.plot(last_raw_f,last_data,c='gray')
    line2, = ax.plot([100,6e9],[0,0])
    line3, = ax.plot([100,6e9],[0,0],'--',c='gray')
    text0 = ax.text(0.02,0.98,'max dB',horizontalalignment='left',verticalalignment='top',fontsize=20,color='b',transform=ax.transAxes)
    text1 = ax.text(0.98,0.98,'TX dB',horizontalalignment='right',verticalalignment='top',fontsize=20,color='r',transform=ax.transAxes)

    def init():
        line0.set_data(last_raw_f,last_data)
        line1.set_data(last_raw_f,last_data)
        ax.set_xlim(4.23e9,4.26e9)
        ax.set_ylim(-100,0)
        return line3, line1, line0, line2, text0, text1

    def animate(i):
        global last_raw_f
        global last_data

        objects = sorted(os.listdir(os.getcwd()))
        delete_files = []
        files_ready = False
        for fm in range(len(objects)):
            if objects[fm][-7:] == '.delete':
                delete_files += [objects[fm][:-7]]
                files_ready = True
        if files_ready:
            line1.set_data(last_raw_f,last_data)
            with h5py.File(delete_files[-1]+'.h5', "r") as fyle:
                raw_VNA = puf.get_raw(fyle)
                TX_amp = fyle["raw_data0/A_RX2"].attrs.get('ampl')[0]
                f0 = fyle["raw_data0/A_RX2"].attrs.get('freq')
                f1 = fyle["raw_data0/A_RX2"].attrs.get('chirp_f')
                LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
            last_raw_f = np.arange(LO+f0, LO+f1, (f1-f0)/len(raw_VNA[:,0]))
            last_data = 20*np.log10(abs(raw_VNA))
            line0.set_data(last_raw_f,last_data)
            line2.set_data([100,6e9],np.array([1,1])*np.percentile(last_data,95))
            text0.set_text(str(round(np.percentile(last_data,95),3))+' dB')
            TX_pow = -14+20*np.log10(TX_amp)
            text1.set_text(str(round(TX_pow,3))+' dBm')
            for delete_file in delete_files:
                os.remove(delete_file+'.delete')
                print 'deleted '+delete_file+'.delete'
                os.remove(delete_file+'.h5')
                print 'deleted '+delete_file+'.h5'

        return line3, line1, line0, line2, text0, text1

    anim = animation.FuncAnimation(fig,animate,init_func=init,frames=None,interval=10,blit=True)
    plt.show()
