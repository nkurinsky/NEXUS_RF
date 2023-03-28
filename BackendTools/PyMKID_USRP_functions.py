import os, sys
import time
import glob
import h5py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import ResonanceFitter as fitres


# import PyMKID_resolution_functions as prf

#print fyle["raw_data0/A_RX2"].attrs.keys()
#print fyle["raw_data0/A_RX2"].attrs.values()


def get_latest_file(text,text2=''):
    list_of_files = glob.glob('*' + text + '*' + text2)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def template(filename,time_threshold=20e-3,ythreshold=0.01,left_time=2e-3,right_time=28e-3,pulse_width=20e-6,osmond=False,period=None):
    trigNum=0
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    with h5py.File(filename, "r") as fyle:
        raw_noise = get_raw(fyle)
        amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
        search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
        decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

    eff_rate = rate/decimation

    time_correction = 1/eff_rate
    left_len = int(left_time*eff_rate)
    right_len = int(right_time*eff_rate)
    pulse_len = int(pulse_width*eff_rate)
    total_len = left_len + right_len
    xthreshold = int(time_threshold*eff_rate)
    all_channels = np.mean(abs(raw_noise-np.mean(raw_noise,axis=0)),axis=1)

    temp_time = np.array(range(left_len+right_len))*time_correction

    if True:
        plt.figure(1)
        plt.plot(np.array(range(len(all_channels)))*time_correction,all_channels, label="stuff")
        plt.axhline(y=ythreshold,ls='--',c='gray')
        plt.axvline(x=max(xthreshold,left_len)*time_correction,ls='--',c='gray')
        plt.axvline(x=(len(raw_noise)-right_len)*time_correction,ls='--',c='gray')
        plt.xlabel("time [s]")
        plt.legend()
        plt.show()

    if period != None:
        time_array = np.array(range(len(all_channels)))*time_correction
        temp_time = time_array[0:int(period*eff_rate)]
        trigNum = int(max(time_array)/period)
        for xx in range(trigNum):
            if xx==0:
                temp_array = raw_noise[0:int(period*eff_rate)]
            else:
                temp_array += raw_noise[xx*int(period*eff_rate):(xx+1)*int(period*eff_rate)]
    elif osmond:
        window_size = 5000
        xx = xthreshold + int(window_size/2) - 1
        window_mean = np.mean(all_channels[xthreshold:xthreshold+window_size])
        window_var = np.var(all_channels[xthreshold:xthreshold+window_size])
        actual_trigNum = 0
        while actual_trigNum == 0:
            if all_channels[xx] > window_mean + 6*np.sqrt(window_var):
                actual_trigNum = 1
                temp_array = raw_noise[xx-left_len:xx+right_len]
                # plt.plot(np.array(range(len(all_channels)))*time_correction,all_channels, label="stuff")
                # plt.figure()
                # plt.plot(np.array(range(xx-left_len,xx+right_len))*time_correction,all_channels[xx-left_len:xx+right_len])
                # plt.show()
                print( 'found first trigger')
                break
            first = all_channels[xx-int(window_size/2)+1]
            last = all_channels[xx+int(window_size/2)+1]
            window_var -= (first-window_mean)**2/window_size
            window_var += (last-window_mean)**2/window_size
            window_mean -= first/window_size
            window_mean += last/window_size
            xx += 1
            if xx == len(all_channels)/10:
                print( 'no  first trigger found')


        xx += int(100e-3*eff_rate)
        while actual_trigNum != 0 and xx < len(all_channels) - right_len:
            window = all_channels[xx-int(window_size/2):xx+int(window_size/2)]
            window_mean = np.mean(window)
            window_std = np.std(window)
            # plt.plot(range(len(window)),window)
            # plt.show()
            peaks = np.argwhere(window > window_mean + 6*window_std)
            # print peaks[0]
            if len(peaks)<2:
                xx += int(100e-3*eff_rate)
                print( 'not enough peaks found')
                continue
            xx = peaks[0][0] + xx - int(window_size/2)
            if actual_trigNum > 9: # only want the second second, according to Taylor
                trigNum += 1
                actual_trigNum += 1
                temp_array += raw_noise[xx-left_len:xx+right_len]
            else:
                actual_trigNum += 1
            xx += int(100e-3*eff_rate)

    else:
        for xx in range(len(all_channels)):
            if xx > max(xthreshold,left_len) and xx < (len(all_channels)-right_len):
                if all_channels[xx] > ythreshold:
                    trigNum += 1
                    if trigNum == 1:
                        temp_array = raw_noise[xx-left_len:xx+right_len]
                    else:
                        temp_array += raw_noise[xx-left_len:xx+right_len]
                    all_channels[xx-left_len:xx+right_len] = 0


    print( "Found %d triggering events"%(trigNum))

    return trigNum, temp_array, temp_time, search_freqs

def vna_file_fit(filename,pickedres,show=False,save=False,verbose=False):
    pickedres = np.array(pickedres)
    VNA_f, VNA_z = read_vna(filename, decimation=1)
    VNA_f = VNA_f*1e-3 #VNA_f in units of GHz after this line

    df = VNA_f[1] - VNA_f[0]
    frequency_range = 1e-3
    index_range = int(frequency_range / df / 2)

    frs = np.zeros(len(pickedres))
    Qrs = np.zeros(len(pickedres))
    Qcs = np.zeros(len(pickedres))
    for MKIDnum in range(len(pickedres)):
        # window_f = (pickedres[MKIDnum] + 10*pickedres[MKIDnum]/3e5)
        # print(VNA_f, window_f)
        # window_index = np.argmin(abs(VNA_f-window_f))
        MKID_index = np.argmin(abs(VNA_f-pickedres[MKIDnum]))
        # index_range = window_index-MKID_index
        MKID_f = VNA_f[max(MKID_index-index_range,0):min(MKID_index+index_range,len(VNA_f))]
        MKID_z = VNA_z[max(MKID_index-index_range,0):min(MKID_index+index_range,len(VNA_f))]
        # frs[MKIDnum], Qrs[MKIDnum], Qc_hat, a, phi, tau, Qc = fitres.finefit(MKID_f, MKID_z, pickedres[MKIDnum])
        res_pars, res_errs = fitres.finefit(MKID_f, MKID_z, pickedres[MKIDnum], restrict_fit_MHz=None, plot=False, verbose=verbose)
        frs[MKIDnum] = res_pars["f0"]
        Qrs[MKIDnum] = res_pars["Qr"]
        Qc_hat = res_pars["QcHat"]
        a = res_pars["zOff"]
        phi = res_pars["phi"]
        tau = res_pars["tau"]
        Qcs[MKIDnum] = res_pars["Qc"]

        fit_z = fitres.resfunc3(MKID_f, frs[MKIDnum], Qrs[MKIDnum], Qc_hat, a, phi, tau)
        #MKID_z_corrected = 1-((1-MKID_z/(a*np.exp(-2j*np.pi*(MKID_f-frs[MKIDnum])*tau)))*(np.cos(phi)/np.exp(1j*phi)))
        #fit_z_corrected = 1-(Qrs[MKIDnum]/Qc)/(1+2j*Qrs[MKIDnum]*(MKID_f-frs[MKIDnum])/frs[MKIDnum])
        fr_idx = find_closest(MKID_f,frs[MKIDnum])

        if show:
            axV = plt.figure().gca()
            axV.plot(MKID_z.real,MKID_z.imag,ls='',marker='.')
            axV.plot(fit_z.real,fit_z.imag,color='lightgrey')
            axV.plot(fit_z[fr_idx].real,fit_z[fr_idx].imag,marker='*')
            axV.plot(MKID_f,20*np.log10(abs(MKID_z)))
            axV.plot(MKID_f,20*np.log10(abs(fit_z)))

            axV.set_aspect('equal', adjustable='box')
            axV.set_xlabel('ADC units')
            axV.set_ylabel('ADC units')
            axV.axvline(x=0, color='gray')
            axV.axhline(y=0, color='gray')
        if save:
            plt.savefig(filename[:-3]+'_res'+str(MKIDnum)+'.png')
            plt.close()

    return frs, Qrs, Qc_hat, a, phi, tau, Qcs
    # return frs, Qrs, Qcs

def get_raw(openfile):
    try:
        raw_data = np.array(openfile["raw_data0/B_RX2/data"])
    except:
        raw_data = np.array(openfile["raw_data0/A_RX2/data"])
    return np.transpose(raw_data)

def clean_noi(file):
    with h5py.File(file, "r") as fyle:
        cleaned_data = np.array(fyle["cleaned_data"])
        data_info = {}
        # print(fyle['radius cleaning coefficient'])
        # print(fyle.keys())
        data_info['sampling_rate'] = fyle["sampling_rate"][()]
        data_info['radius'] = fyle["radius"][()]
        data_info['angle'] = fyle['angle'][()]
        data_info['radius cleaning coefficient'] = {}
        data_info['arc cleaning coefficient'] = {}
        for key in fyle['radius cleaning coefficient'].keys():
            data_info['radius cleaning coefficient'][float(key)] = fyle['radius cleaning coefficient'][key][()]
            data_info['arc cleaning coefficient'][float(key)] = fyle['arc cleaning coefficient'][key][()]
    return cleaned_data, data_info

def read_vna(filename, decimation=1,verbose=False):
    #Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    try:
        with h5py.File(filename, "r") as fyle:
            raw_VNA = get_raw(fyle)
            amplitude = fyle["raw_data0/B_RX2"].attrs.get('ampl')
            rate = fyle["raw_data0/B_RX2"].attrs.get('rate')
            LO = fyle["raw_data0/B_RX2"].attrs.get('rf')
            f0 = fyle["raw_data0/B_RX2"].attrs.get('freq')[0]
            f1 = fyle["raw_data0/B_RX2"].attrs.get('chirp_f')[0]
            delay = (fyle["raw_data0/B_RX2"].attrs.get('delay')-1)*1e9 # ns
    except:
        with h5py.File(filename, "r") as fyle:
            raw_VNA = get_raw(fyle)
            amplitude = fyle["raw_data0/A_RX2"].attrs.get('ampl')
            rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
            LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
            f0 = fyle["raw_data0/A_RX2"].attrs.get('freq')[0]
            f1 = fyle["raw_data0/A_RX2"].attrs.get('chirp_f')[0]
            delay = (fyle["raw_data0/A_RX2"].attrs.get('delay')-1)*1e9 # ns
    eff_rate = rate/decimation

    if verbose:
        print( "\n\nData taken "+str(Dt_tm))
        print( "Reported LO is "+str(LO*1e-6)+" MHz")
        print( "Reported rate is %f MHz"%(rate/1e6))
        print( "Entered decimation is %d"%(decimation))
        print( "\tEffective rate is %f kHz"%(eff_rate/1e3))
        print( "Reported amplitudes are "+str(amplitude))
        print( "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm")

    raw_f = np.linspace(LO+f0,LO+f1,len(raw_VNA[:,0]))*1e-6

    return raw_f, raw_VNA[:,0]#, amplitude

def avg_noi(filename,time_threshold=0.05,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    try:
        with h5py.File(filename, "r") as fyle:
            raw_noise = get_raw(fyle)
            amplitude = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
            rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
            LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
            search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
            decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')
    except:
        with h5py.File(filename, "r") as fyle:
            raw_noise = get_raw(fyle)
            amplitude = fyle["raw_data0/B_TXRX"].attrs.get('ampl')
            rate = fyle["raw_data0/B_RX2"].attrs.get('rate')
            LO = fyle["raw_data0/B_RX2"].attrs.get('rf')
            search_freqs = fyle["raw_data0/B_RX2"].attrs.get('rf') + fyle["raw_data0/B_RX2"].attrs.get('freq')
            decimation = fyle["raw_data0/B_RX2"].attrs.get('decim')

    eff_rate = rate/decimation

    if verbose:
        print( "\n\nData taken "+str(Dt_tm))
        print( "Reported LO is "+str(LO*1e-6)+" MHz")
        print( "Reported rate is %f MHz"%(rate/1e6))
        print( "Reported decimation is %d"%(decimation))
        print( "\tEffective rate is %f kHz"%(eff_rate/1e3))
        print( "Reported amplitudes are "+str(amplitude))
        print( "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm")
        print( "Tones are "+str(search_freqs*1e-6)+" MHz")

    time_correction = 1/eff_rate

    time_array = time_correction*np.arange(0,len(raw_noise))

    array_mean = np.mean(raw_noise[time_array>time_threshold,:], axis=0,dtype=complex)

    return search_freqs*1e-6, array_mean

def unavg_noi(filename,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    data_info = {}

    try:
        with h5py.File(filename, "r") as fyle:
            raw_noise = get_raw(fyle)
            amplitudes = fyle["raw_data0/B_TXRX"].attrs.get('ampl')
            tx_gain = fyle["raw_data0/B_TXRX"].attrs.get('gain')
            rate = fyle["raw_data0/B_RX2"].attrs.get('rate')
            LO = fyle["raw_data0/B_RX2"].attrs.get('rf')
            search_freqs = fyle["raw_data0/B_RX2"].attrs.get('rf') + fyle["raw_data0/B_RX2"].attrs.get('freq')
            decimation = fyle["raw_data0/B_RX2"].attrs.get('decim')
    except:
        with h5py.File(filename, "r") as fyle:
            raw_noise = get_raw(fyle)
            amplitudes = fyle["raw_data0/A_TXRX"].attrs.get('ampl')
            tx_gain = fyle["raw_data0/A_TXRX"].attrs.get('gain')
            rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
            LO = fyle["raw_data0/A_RX2"].attrs.get('rf')
            search_freqs = fyle["raw_data0/A_RX2"].attrs.get('rf') + fyle["raw_data0/A_RX2"].attrs.get('freq')
            decimation = fyle["raw_data0/A_RX2"].attrs.get('decim')

    data_info['rate'] = rate
    data_info['LO'] = LO
    data_info['search freqs'] = search_freqs*1e-6
    data_info['decimation'] = decimation

    full_power = -14
    powers = tx_gain + full_power + 20*np.log10(amplitudes)

    data_info['powers'] = powers

    eff_rate = rate/decimation
    if verbose:
        print( "\n\nData taken "+str(Dt_tm))
        print( "Reported LO is "+str(LO*1e-6)+" MHz")
        print( "Reported rate is %f MHz"%(rate/1e6))
        print( "Reported decimation is %d"%(decimation))
        print( "\tEffective rate is %f kHz"%(eff_rate/1e3))
        print( "Reported amplitudes are "+str(amplitude))
        print( "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm")
        print( "Tones are "+str(search_freqs*1e-6)+" MHz")

    total_idx = len(raw_noise)
    total_time = total_idx/eff_rate


    time_correction = 1/eff_rate # s?
    time = np.linspace(time_correction,total_time,total_idx)

    data_info['time'] = time
    data_info['sampling period'] = time_correction

    return raw_noise, data_info

def avg_VNA(filename, decimation=1, f0=None, f1=None, targets=None,verbose=False):
    Dt_tm = filename.split('.')[0].split('_')[2] + '_' + filename.split('.')[0].split('_')[3]

    with h5py.File(filename, "r") as fyle:
        raw_VNA = get_raw(fyle)
        time = fyle["raw_data0/A_RX2"].attrs.get('chirp_t')[0]
        amplitude = fyle["raw_data0/A_RX2"].attrs.get('ampl')
        rate = fyle["raw_data0/A_RX2"].attrs.get('rate')
        LO = fyle["raw_data0/A_RX2"].attrs.get('rf')

    eff_rate = rate/decimation
    if verbose:
        print( "\n\nData taken "+str(Dt_tm))
        print( "Reported LO is "+str(LO*1e-6)+" MHz")
        print( "Reported rate is %f MHz"%(rate/1e6))
        print( "Entered decimation is %d"%(decimation))
        print( "\tEffective rate is %f kHz"%(eff_rate/1e3))
        print( "Reported amplitudes are "+str(amplitude))
        print( "\tPowers are "+str(-11+20*np.log10(amplitude))+" dBm")

    raw_f = np.arange(f0, f1, (f1-f0)/len(raw_VNA[:,0]))

    array_mean = np.array([])
    for freq in targets:
        print( str(raw_f[np.argmin(abs(raw_f-freq))])+' MHz')
        array_mean = np.append(array_mean,raw_VNA[np.argmin(abs(raw_f-freq))])

    return targets, array_mean, Dt_tm, int(time*rate/len(raw_VNA)), time/len(raw_VNA)

def find_closest(vector,value):
    return np.argmin(abs(vector-value))

def fit_poly(x,y,order=1):
    A = np.zeros((len(x),order+1))
    b = y
    for power in range(order+1):
        A[:,power] = np.ones((len(x),1)) * x ** power
    coefficients = np.linalg.solve(np.matmul(A.T,A),np.matmul(A.T,b))
    return coefficients

def plot_VNA(filename, fig_obj1=None, fig_obj2=None):
    f, z = read_vna(filename)
    crop = 1000
    f = f[crop:-crop]*1e-3
    z = z[crop:-crop]
    f = f[::5]
    z = z[::5]

    resonances, _, _, _, _, _, _ = vna_file_fit(filename,[4.24218])#[3.468, 3.486, 3.503, 3.505, 3.516, 3.527, 3.539])
    near = .0007
    near_res = []
    for resonance in resonances:
        near_this_res = np.logical_and(f > resonance - near, f < resonance + near)
        if len(near_res) == 0:
            near_res = near_this_res
        else:
            near_res = np.logical_or(near_res,near_this_res)
    
    if fig_obj1 is None:
        fig_obj1 = plt.figure(figsize=(3,3),dpi=300)
    ax0 = fig_obj1.gca()
    ax0.plot(np.real(z[near_res]),np.imag(z[near_res]),color = 'red',marker ='.',linestyle='',markersize=2)
    ax0.plot(np.real(z[np.logical_not(near_res)]),np.imag(z[np.logical_not(near_res)]),marker = '.',linestyle = '',markersize=2)
    ax0.set_title(filename)

    if fig_obj2 is None:
        fig_obj2 = plt.figure(figsize=(3,3),dpi=300)
    ax1 = fig_obj1.gca()
    ax1.plot(f[near_res], 20*np.log10(abs(z[near_res])),color = 'red',marker ='.',linestyle='',markersize=2)
    ax1.plot(f[np.logical_not(near_res)],20*np.log10(abs(z[np.logical_not(near_res)])),marker='.',linestyle='',markersize=2)
    ax1.title(filename)
    # plt.show()

def plot_noise_and_vna(noise,VNA_z,fit_z=None,f_idx=None,char_zs=None,alpha=0.1,title='',show_legend=True,fig_obj=None,save_directory=None,verbose=True):
    if fig_obj == None:
        fig = plt.figure('noise and vna ' + title,figsize=(3,3),dpi=300)
    else:
        fig = fig_obj
    fig.gca();

    plt.title(title + ' complex $S_{21}$')

    ## Plot the VNA result in complex S21
    plt.plot(VNA_z.real,VNA_z.imag,'k',ls='',marker='.',label='VNA',markersize=1)

    ## Plot the noise points
    plt.plot(noise.real,noise.imag,alpha=alpha,marker='.',ls='',label='noise timestream')
    

    if type(char_zs) != type(None):
        plt.plot(char_zs.real,char_zs.imag,\
                 marker='.',ls='--',markersize=10,color='y',label='calibration timestreams')
    if type(fit_z) != type(None):
        plt.plot(fit_z.real,fit_z.imag,ls='-',color='r',label='resonance fit')
    
    if f_idx:
        plt.plot(VNA_z[f_idx].real,VNA_z[f_idx].imag,\
                 ls = '',marker='.',color='r',markersize=10,label='VNA closest to readout frequency')

    real_mean = np.mean(noise.real,axis=0,keepdims=True)
    imag_mean = np.mean(noise.imag,axis=0,keepdims=True)

    if title[-5:] != 'ideal':
        radius_mean = np.sqrt(real_mean**2 + imag_mean**2)

        angles = np.angle(real_mean + 1j*imag_mean)

        dtheta = 0.2

        for idx in range(len(angles)):
            theta_plot = np.linspace(angles[idx] - dtheta,angles[idx] + dtheta, 100)
            x_plot = radius_mean[idx]*np.cos(theta_plot)
            y_plot = radius_mean[idx]*np.sin(theta_plot)
            plt.plot(x_plot,y_plot,color='k',alpha = 0.6,label='arc length direction')

            radius_plot = np.linspace(0.95*radius_mean[idx],1.05*radius_mean[idx],100)
            x_plot = radius_plot*np.cos(angles[idx])
            y_plot = radius_plot*np.sin(angles[idx])
            plt.plot(x_plot,y_plot,ls='-.',color='k',alpha=0.6,label='radius direction')

    plt.plot(real_mean,imag_mean,'g',markersize=10,marker='*',ls='',label='timestream average')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('ADC units')
    plt.ylabel('ADC units')
    plt.axvline(x=0, color='gray')
    plt.axhline(y=0, color='gray')
    if show_legend:
        plt.legend(loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    if type(save_directory) != type(None):
        plt.tight_layout()
        plt.savefig(save_directory + title + ' noise and vna.png')

    return fig


if __name__ == '__main__':
    plot_VNA('USRP_VNA_20200428_123007.h5')
