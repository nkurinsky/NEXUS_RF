from __future__ import division
import os
import numpy.fft as fft
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatchesss
import h5py
import PyMKID_USRP_functions as PUf
import scipy
from scipy.signal import periodogram,get_window,coherence,welch,blackman
import ResonanceFitter as fitres
import MB_equations as MBe
import MB_analysis as MBa
# reload(PUf)
# import settings

#import scipy.signal as sig
#import time



c_wheel_0 = ['C0','C1','C2','C3','C4','C5','C6','C8','C9','C7']
c_wheel_1 = ['deepskyblue','sandybrown','lightgreen','lightcoral','mediumorchid','peru','lightpink','khaki','paleturquoise','silver']

def discrete_FT(data,axis=0):
    return (1./len(data))*fft.fft(data,axis=axis)

def discrete_IFT(data,axis=0):
    return (len(data))*fft.ifft(data,axis=axis)

def electronics_basis(noise_timestream,axis_option=0):
    """
    input: a noise timestream that comes from the a "USRP_Noise_*" file
            S21 value in complex form
    output: timestreams of the radius and arc length directions
    """
    if axis_option == 'multiple freqs':
        ndims = noise_timestream.ndim
        axis_average = tuple(range(ndims-1))
    elif axis_option == 0:
        axis_average = 0
    # print(noise_timestream)

    ## This takes the magnitude of a+ib --> sqrt(a*a + b*b)
    full_radius_timestream = abs(noise_timestream)
    # print(full_radius_timestream)

    ## Now we find the mean radius direction value for the full timestream from the magnitude of S21
    radius = np.mean(full_radius_timestream,axis=axis_average,dtype=np.float64)
    # print(full_radius_timestream.shape,axis_average)
    # print(radius)

    ## Subtract off the average
    radius_timestream = full_radius_timestream - radius

    ## Find the mean of the baseline subtraced noise 
    mean = np.mean(noise_timestream,axis=axis_average,dtype=complex)

    ## Take the arctan2 of the complex mean
    angle = np.angle(mean)

    ## Apply a phase rotation to the noise timestram
    noise_timestream_rotated = noise_timestream*np.exp(-1j*angle)

    ## Pull the phase angle of the time stream
    angle_timestream = np.angle(noise_timestream_rotated)

    ## Convert from an angle to an arc length
    arc_length_timestream = angle_timestream*radius

    ## Return the radial direction timestream, arclength direction time stream, radius average, and phase average
    return radius_timestream, arc_length_timestream, radius, angle

def resonator_basis(noise_timestream,readout_f,VNA_f,VNA_z,char_f,char_z):

    def rotate_to_ideal(z,f,fr,a,tau,phi):
        return 1-((1-z/(a*np.exp(-2j*np.pi*(f-fr)*tau)))*(np.cos(phi)/np.exp(1j*phi)))

    char_res_idx = int(len(char_f)/2)
    # Define region around this resonance, and fit for parameters
    index_range = 4000
    readout_index = PUf.find_closest(VNA_f,readout_f)
    MKID_f = VNA_f[max(readout_index-index_range,0):min(readout_index+index_range,len(VNA_f))]
    MKID_z = VNA_z[max(readout_index-index_range,0):min(readout_index+index_range,len(VNA_f))]
    fine_pars, fine_errs = fitres.finefit(MKID_f, MKID_z, readout_f)

    ## Unpack the dictionaries from fine fit result
    fit_fr = fine_pars["f0"]
    fit_Qr = fine_pars["Qr"]
    fit_Qc_hat = fine_pars["QcHat"]
    fit_a   = fine_pars["zOff"]
    fit_phi = fine_pars["phi"]
    fit_tau = fine_pars["tau"]
    fit_Qc  = fine_pars["Qc"]

    fit_Qi = fit_Qr*fit_Qc/(fit_Qc-fit_Qr)
    # print(fit_Qr,fit_Qc)


    # Transform VNA to ideal space
    MKID_z_ideal = rotate_to_ideal(MKID_z,MKID_f,fit_fr,fit_a,fit_tau,fit_phi)

    # Transform the VNA fit to ideal space
    fit_f = np.linspace(MKID_f[0],MKID_f[-1],10000)
    fit_z = fitres.resfunc3(fit_f, fit_fr, fit_Qr, fit_Qc_hat, fit_a, fit_phi, fit_tau)
    fit_z_ideal = rotate_to_ideal(fit_z,fit_f,fit_fr,fit_a,fit_tau,fit_phi)

    # find some important indices in the f vector
    first_char_fit_idx = PUf.find_closest(fit_f,char_f[0])
    last_char_fit_idx = PUf.find_closest(fit_f,char_f[-1])
    # print(first_char_fit_idx)
    # print(last_char_fit_idx)
    res_f_idx = PUf.find_closest(MKID_f,readout_f)
    res_fit_idx = PUf.find_closest(fit_f,readout_f)

    # Find VNA-data subset that covers the characterization-data region in frequency space
    char_region_f = fit_f[first_char_fit_idx:last_char_fit_idx]
    char_region_z = fit_z[first_char_fit_idx:last_char_fit_idx]
    char_region_z_ideal = fit_z_ideal[first_char_fit_idx:last_char_fit_idx]

    # Get the angle (in complex space) of the characterization data
    real_fit = np.polyfit(char_f,char_z.real,1)
    imag_fit = np.polyfit(char_f,char_z.imag,1)
    char_angle = np.angle((real_fit[0]*(char_region_f[-1]-char_region_f[0]))+1j*(imag_fit[0]*(char_region_f[-1]-char_region_f[0])))

    # Get the angle (in complex space) of the VNA data
    real_fit = np.polyfit(char_region_f,char_region_z.real,1)
    imag_fit = np.polyfit(char_region_f,char_region_z.imag,1)
    char_region_angle = np.angle((real_fit[0]*(char_region_f[-1]-char_region_f[0]))+1j*(imag_fit[0]*(char_region_f[-1]-char_region_f[0])))

    # Get the angle (in complex ideal space) of the VNA data
    real_fit = np.polyfit(char_region_f,char_region_z_ideal.real,1)
    imag_fit = np.polyfit(char_region_f,char_region_z_ideal.imag,1)
    char_region_ideal_angle = np.angle((real_fit[0]*(char_region_f[-1]-char_region_f[0]))+1j*(imag_fit[0]*(char_region_f[-1]-char_region_f[0])))

    # Rotate characterization data to VNA data
    char_z_rotated = (char_z-char_z[char_res_idx])\
                     *np.exp(1j*(-1*char_angle+char_region_angle))\
                     +fit_z[res_fit_idx]

    # print(char_z_rotated,char_f,fit_fr,fit_a,fit_tau,fit_phi)
    char_z_rotated_ideal = rotate_to_ideal(char_z_rotated,char_f,fit_fr,fit_a,fit_tau,fit_phi)

    timestream_rotated = (noise_timestream-np.mean(noise_timestream,dtype=complex))\
                      *np.exp(1j*(-1*char_angle+char_region_angle))\
                      +fit_z[res_fit_idx]

    timestream_rotated_ideal = rotate_to_ideal(timestream_rotated,readout_f,fit_fr,fit_a,fit_tau,fit_phi)
    dS21 = timestream_rotated_ideal-np.mean(timestream_rotated_ideal,dtype=complex)

    # Extra rotation for imaginary direction --> frequency tangent direction
    # This is necessary if data_freqs[0] != primary_fr
    # Note that char_f[res_idx] == data_freqs[0] should always be true
    # data_freqs[0] != primary_fr means our vna fit before data taking gave a different fr than our vna fit now

    phase_adjust = 1
    # phase_adjust = np.exp(1j*(np.sign(char_region_ideal_angle)*0.5*np.pi-char_region_ideal_angle))

    frequency = phase_adjust*dS21.imag*fit_Qc/(2*fit_Qr**2)
    dissipation = phase_adjust*dS21.real*fit_Qc/(fit_Qr**2)
    # print(type(frequency[2]),type(dissipation[2]),type(fit_Qc))

    ideal = {}
    ideal['f'] = MKID_f
    ideal['z'] = MKID_z_ideal
    ideal['char z'] = char_z_rotated_ideal
    ideal['timestream'] = timestream_rotated_ideal
    ideal['fit f'] = fit_f
    ideal['fit z'] = fit_z_ideal

    resonator = {}
    resonator['fr'] = fit_fr
    resonator['Qr'] = fit_Qr
    resonator['Qc'] = fit_Qc
    resonator['a'] = fit_a

    if False:
        # PUf.plot_noise_and_vna(noise_timestream,MKID_z,f_idx=res_f_idx,char_zs=char_z,title='S21')

        PUf.plot_noise_and_vna(timestream_rotated_ideal,MKID_z_ideal,\
                               f_idx=res_f_idx,char_zs=char_z_rotated_ideal,title='ideal space')
        plt.show(False)
        # plt.pause(5)
        raw_input('press enter to close all plots')
        plt.close('all')

        # plt.figure(2)
        # fit_fr_idx = PUf.find_closest(MKID_f,fit_fr)
        # # Plotting the corrected data (moved to ideal S21 space)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.axvline(x=0,c='k')
        # plt.axhline(y=0,c='k')
        # plt.plot(MKID_z_ideal.real,MKID_z_ideal.imag,label='ideal space vna')
        # plt.plot(char_z_rotated_ideal.real,char_z_rotated_ideal.imag,label='calibrated calibration data')
        # plt.plot(MKID_z_ideal[res_f_idx].real,MKID_z_ideal[res_f_idx].imag,'o',label='vna closest to readout frequency')
        # plt.plot(MKID_z_ideal[fit_fr_idx].real,MKID_z_ideal[fit_fr_idx].imag,'o',label='vna closest to new fr')
        # plt.plot(timestream_rotated_ideal[::10].real,timestream_rotated_ideal[::10].imag,ls='',marker='.')
        # # plt.legend()
        # plt.show()

    return frequency, dissipation, ideal, resonator

def quasiparticle_basis(frequency,dissipation,data_T,MB_results,readout_f):

    MB_f0 = MB_results[0]*1e3
    MB_Delta = MB_results[1] # meV
    MB_alpha = MB_results[2]
    MB_Qi0 = MB_results[3]

    k1 = MBe.kappa_1(data_T, readout_f*1e6, MB_Delta*1e-3)*1e6*1e6*1e6 # um^3
    k2 = MBe.kappa_2(data_T, readout_f*1e6, MB_Delta*1e-3)*1e6*1e6*1e6
    nqp_theta = (0.5*np.pi-np.arctan(k2/k1))/np.pi*180
    # print(nqp_theta)


    dnqp_k1 = dissipation/(MB_alpha*k1)
    dnqp_k2 = 2*frequency/(MB_alpha*k2)

    return dnqp_k1, dnqp_k2

def PyMKID_decimate(timestream, decimation,axis=0):
    """
    in house decimate function that iterates over scipy's decimate function

    NB: scipy's decimate function by default uses an IIR low pass filter and
    then downsamples by the requested amount. Scipy recommends not using it for
    downsampling greater than 13, so this function iterates over scipy's decimate
    function in factors of 10

    input: timestream array, decimation factor (needs to be a multiple of 10)
    output: timestream array that is smaller in size by the decimation factor
    """
    if decimation <= 10:
        return decimate(timestream,int(decimation),axis=axis)
    elif decimation % 10 == 0:
        timestream_decimated = PyMKID_decimate(timestream,int(decimation/10))
        return decimate(timestream_decimated,10,axis=axis)
    else:
        raise Exception('decimation must be divisible by the highest power of 10 less than the desired decimation')

def average_decimate(timestream,decimation):
    '''
    decimation code from alvaro's "noise_removal" function

    input: timestream (I think it needs to be real), decimation factor
    output: decimated timestream
    '''
    # ids = np.arange(len(timestream))//decimation
    # timestream_decimated = np.bincount(ids,timestream)/np.bincount(ids)
    dtype = timestream.dtype
    decimation_len = int(len(timestream)/decimation)
    chunked_timestream = create_chunks(timestream,decimation_len)
    timestream_decimated=np.mean(chunked_timestream,axis=0,dtype=dtype)
    return timestream_decimated

def CSD_avg(data_array,time_correction,avg_num,fraction_of_data=0.5,rejects=np.array([])):
    start_idx = int((1-fraction_of_data)*len(data_array))
    rejects[rejects>start_idx]
    chunk_len = int(fraction_of_data*len(data_array)/avg_num)
    freqs = fft.fftfreq(chunk_len,d=time_correction)
    CSD_avg = (1+1j)*np.zeros((len(data_array.T),len(data_array.T),chunk_len))
    actual_avg = avg_num - len(rejects)
    for Nm in range(len(data_array.T)):
        for Nn in range(len(data_array.T)):
            last_chunk_has_pulse = False
            for Am in range(avg_num):
                Am_start = int(start_idx+chunk_len*Am)
                Am_stop = int(start_idx+chunk_len*(Am+1))

                if any(np.logical_and(rejects >= Am_start,rejects < Am_stop)):
                    last_chunk_has_pulse = True
                    continue
                elif last_chunk_has_pulse:
                    last_chunk_has_pulse = False
                    continue
                else:
                    timestream_1 = data_array[Am_start:Am_stop,Nm]
                    timestream_2 = data_array[Am_start:Am_stop,Nn]
                    print(Am_start,Am_stop)
                    total_time = time_correction*chunk_len
                    CSD_avg[Nm,Nn] += (1/actual_avg)\
                                    *CSD(total_time,timestream_1,timestream_2)

    return freqs, CSD_avg

def CSD(total_time,timestream_1,timestream_2,axis=0):
    CSD = total_time*\
          np.conj(discrete_FT(timestream_1,axis=axis))*\
          discrete_FT(timestream_2,axis=axis)
    return CSD

def a_estimator(s,d,rejects=[]):
    """estimates amplitude of template given signal model s and raw data d"""
    return sum(s*d)/sum((s**2))

def noise_removal(coherent_data,removal_decimation=1,rejects=[],verbose=False):

    #Transpose data to work with templates if needed:
    coherent_data = np.transpose(coherent_data)

    coherent_data_clean = np.zeros(coherent_data.shape)

    coefficients = []
    #-----go through the templates and compute clean data-----
    for t in range(len(coherent_data)): #loop over all tones
        if verbose:
            print('working on tone {}, time is {}'.format(t,datetime.now()))

        #build template with undecimated data
        if t == 0:
            off_tone_data = np.delete(coherent_data,t,axis=0) #delete tone for which template is being created #len = tones-1
        else:
            off_tone_data = np.delete(coherent_data,[0, t],axis=0)

        print(off_tone_data.shape)

        ## Take the mean of all other tones at every sample #len = number of samples in data_noise
        template = np.mean(off_tone_data,axis=0,dtype=np.float64) 
        # print(template.shape)

        ## Pull a z score from the template
        template_zscores =  stats.zscore(template)

        ## Rescales to max = 1 and mean~0
        template_norm = template_zscores/np.max(template_zscores) 
        # print(np.max(stats.zscore(template)))
        template_ratio = template_norm/template
        # print(template_ratio[0:10])
        norm_scaling = 1/np.std(template)/np.max(template_zscores)

        ## Decimate template and data to get coeff
        template_decimated = average_decimate(template_norm,removal_decimation)
        # print(template_decimated.shape)
        coherent_data_decimated = average_decimate(coherent_data[t],removal_decimation)
        # print(coherent_data_decimated.shape)

        a1 = a_estimator(template_decimated,coherent_data_decimated) #compute the amplitude coefficient
        a1_not_norm = a_estimator(template,coherent_data[t])

        ## Clean undecimated data
        coherent_data_clean[t] = coherent_data[t] - a1*template_norm

        coefficients_tone={}
        coefficients_tone['normalized'] = a1*norm_scaling
        coefficients_tone['not normalized'] = a1
        coefficients.append(coefficients_tone)


    return np.transpose(coherent_data_clean), coefficients

def noise_removal_simple(timestream_1,timestream_2,transmission_1,transmission_2):
    cleaned = np.zeros((len(timestream_1),2))
    coefficient = transmission_1/transmission_2
    cleaned[:,0] = timestream_1 - timestream_2*coefficient
    cleaned[:,1] = timestream_2
    return cleaned, coefficient

def save_clean_timestreams(h5_file,radius,angle,cd1_clean,cd2_clean,fs,cd1_coeff,cd2_coeff,decimation=1,override=False):
    if len(cd1_clean) > 100:
        np.transpose(cd1_clean)
    if len(cd2_clean) > 100:
        np.transpose(cd2_clean)

    data_clean = (radius+cd1_clean)*np.exp(1j*((cd2_clean/radius)+angle))
    # data_clean = (np.mean(abs(data_raw),axis=0,dtype=np.float64)+cd1_clean)*np.exp(1j*((cd2_clean/np.mean(abs(data_raw),axis=0,dtype=np.float64))+np.angle(np.mean(data_raw,axis=0,dtype=complex))))
    # print(cd1_coeff)

    data_clean_average = np.mean(data_clean,dtype='complex128')
    print(abs(data_clean_average))
    print(np.angle(data_clean_average))

    cleaned_filename = h5_file[:-3]+'_cleaned.h5'

    if os.path.exists(cleaned_filename) and override == True:
        os.remove(cleaned_filename)
        print('saving clean_data to {} because override=True!'.format(cleaned_filename))
        with h5py.File(cleaned_filename, 'w') as fyle:
            fyle['cleaned_data'] = data_clean
            fyle['sampling_rate'] = fs
            fyle['radius'] = radius
            fyle['angle'] = angle
            for i in range(len(cd1_coeff)):
                fyle['radius cleaning coefficient/'+str(i)] = cd1_coeff[i]['normalized']
                fyle['arc cleaning coefficient/'+str(i)] = cd2_coeff[i]['normalized']
    elif override == True:
        with h5py.File(cleaned_filename, 'w') as fyle:
            fyle['cleaned_data'] = data_clean
            fyle['sampling_rate'] = fs
            fyle['radius'] = radius
            fyle['angle'] = angle
            for i in range(len(cd1_coeff)):
                fyle['radius cleaning coefficient/'+str(i)] = cd1_coeff[i]['normalized']
                fyle['arc cleaning coefficient/'+str(i)] = cd2_coeff[i]['normalized']


    # with h5py.File(cleaned_filename, 'w') as fyle:
    #     print(fyle.keys())
    #     if 'cleaned_data' in fyle.keys() and 'radius cleaning coefficient' in fyle.keys():
    #         print('cleaned_data already exists! If you set override=False, nothing will happen.')
    #         if override==True:
    #             print('saving clean_data to {} because override=True!'.format(h5_file))
    #             del fyle['cleaned_data']
    #             del fyle['radius cleaning coefficient']
    #             del fyle['arc cleaning coefficient']
    #             fyle['cleaned_data'] = data_clean
    #             for i in range(len(cd1_coeff)):
    #                 fyle['radius cleaning coefficient/'+str(i)] = cd1_coeff[i]['normalized']
    #                 fyle['arc cleaning coefficient/'+str(i)] = cd2_coeff[i]['normalized']
    #     elif 'cleaned_data' in fyle.keys() and not 'radius cleaning coefficient' in fyle.keys():
    #         if 'cleaned_data' in fyle.keys() and 'radius cleaning coefficient' in fyle.keys():
    #             print('cleaned_data already exists! If you set override=False, nothing will happen.')
    #             if override==True:
    #                 print('saving clean_data to {} because override=True!'.format(h5_file))
    #                 del fyle['cleaned_data']
    #                 fyle['cleaned_data'] = data_clean
    #                 for i in range(len(cd1_coeff)):
    #                     fyle['radius cleaning coefficient/'+str(i)] = cd1_coeff[i]['normalized']
    #                     fyle['arc cleaning coefficient/'+str(i)] = cd2_coeff[i]['normalized']
    #     else:
    #         print('saving clean_data to {}!'.format(h5_file))
    #         fyle['cleaned_data'] = data_clean
    #         for i in range(len(cd1_coeff)):
    #             fyle['radius cleaning coefficient/'+str(i)] = cd1_coeff[i]['normalized']
    #             fyle['arc cleaning coefficient/'+str(i)] = cd2_coeff[i]['normalized']

    return data_clean

def create_chunks(timestreams,num_chunks):
    """
    will take a LxN array of L-length timestreams of N frequencies and recast it
    into a 3D array that is (L/num_chunks) x num_chunks x N

    input: timestream is a numpy array, num_chunks is an integer
    output: a 3D array
    """
    dtype = timestreams.dtype
    L = int(timestreams.shape[0])
    if timestreams.ndim == 1:
        timestreams = np.expand_dims(timestreams,axis=1)
    N = int(timestreams.shape[1])
    if L%num_chunks != 0:
        raise Exception('timestream must be divisible into equal sized chunks')
    chunk_L = int(L / num_chunks)
    chunked_timestreams = np.zeros((chunk_L,num_chunks,N),dtype=dtype)
    for freq_idx in range(N):
        timestream = timestreams[:,freq_idx]
        chunked_timestream = np.reshape(timestream,(chunk_L,num_chunks),order='F')

        chunked_timestreams[:,:,freq_idx] = chunked_timestream
    return chunked_timestreams

def identify_bad_chunks(chunked_time,pulse_times):
    bad_chunks = []
    # print(pulse_times,chunked_time)
    chunk_starts = chunked_time[0,:,0]
    for pulse_time in pulse_times:

        chunked_pulse_idx = np.argwhere(pulse_time>chunk_starts)

        chunk_idx = chunked_pulse_idx[-1][0]
        bad_chunks.append(chunk_idx)
    return bad_chunks

def plot_PSDs(f,P_1,P_2,noise_data_file,directions,units,savefig,data_freqs=[0],title='',P_1_clean=None,P_2_clean=None,fig_0=None,axes_0=None):
    # print(axes_0[0,0])
    num_freqs = len(data_freqs)
    if type(fig_0) == type(None):
        fig_0, axes_0 = plt.subplots(2,num_freqs,sharex=True,sharey='row',figsize=(5*num_freqs,10))
    if num_freqs == 1:
        axes_0 = np.expand_dims(axes_0,axis=1)
        P_1 = np.expand_dims(P_1,axis=1)
        P_2 = np.expand_dims(P_2,axis=1)
        if type(P_1_clean) != type(None):
            P_1_clean = np.expand_dims(P_1_clean,axis=1)
            P_2_clean = np.expand_dims(P_2_clean,axis=1)

    # max_yval = max((np.amax(P_1[1:,:]),np.amax(P_2[1:,:])))
    # min_yval = min((np.amin(P_1[1:,:]),np.amin(P_2[1:,:])))
    # ymin = 10**-1*min_yval
    # ymax = 10**1*max_yval
    if savefig == 'electronics':
        ymin = 10**-16
        ymax = 10**-9
    elif savefig == 'nqp':
        ymin = 10**-7
        ymax = 10**2
    elif savefig[:3] == 'res':
        ymin = 10**-22
        ymax = 10**-16
    xmin = 10**0
    xmax = 10**5
    f = f[f > 0]
    for N in range(num_freqs):
        # print(N,axes_0[0,N])
        axes_0[0,N].loglog(f,P_1[1:,N],c=c_wheel_0[N],label=str(round(data_freqs[N],3))+' decimated')
        axes_0[1,N].loglog(f,P_2[1:,N],'--',c=c_wheel_0[N],label=str(round(data_freqs[N],3))+' decimated')
        if type(P_1_clean) != type(None):
            axes_0[0,N].loglog(f,P_1_clean[1:,N],c=c_wheel_1[N],label=str(round(data_freqs[N],3))+' cleaned')
            axes_0[1,N].loglog(f,P_2_clean[1:,N],'--',c=c_wheel_1[N],label=str(round(data_freqs[N],3))+' cleaned')
        axes_0[0,N].legend(prop={'size':15})
        axes_0[0,N].tick_params(labelbottom=True,labelleft=True,labelsize=15)
        axes_0[0,N].set_ylim([ymin,ymax])
        axes_0[0,N].set_xlim([xmin,xmax])
        axes_0[1,N].legend(prop={'size':15})
        axes_0[1,N].tick_params(labelbottom=True,labelleft=True,labelsize=15)
        axes_0[1,N].set_ylim([ymin,ymax])
        axes_0[0,N].set_xlim([xmin,xmax])
        axes_0[0,N].grid(True)
        axes_0[1,N].grid(True)
        if N >= 1:
            axes_0[0,N].set_title('off resonance',fontsize=15)
    axes_0[0,0].legend(prop={'size':15})
    axes_0[0,0].tick_params(labelbottom=True,labelleft=True,labelsize=15)
    axes_0[0,0].set_ylim([ymin,ymax])
    axes_0[0,N].set_xlim([xmin,xmax])
    axes_0[0,0].legend(prop={'size':15})
    axes_0[1,0].tick_params(labelbottom=True,labelleft=True,labelsize=15)
    axes_0[0,0].set_ylim([ymin,ymax])
    axes_0[0,N].set_xlim([xmin,xmax])
    axes_0[0,0].set_title('on resonance',fontsize=15)


    fig_0.add_subplot(211, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.ylabel( directions[0] + ' PSD [(' + units[0] + ')$^2 Hz^{-1}$]',fontsize=15,labelpad=70)

    fig_0.add_subplot(212, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel(directions[1] + ' PSD [(' + units[1] + ')$^2 Hz^{-1}$]',fontsize=15,labelpad=70)
    plt.xlabel('Frequency [Hz]')
    fig_0.suptitle(title)
    # plt.show(False)
    # raw_input('press enter to close all plots')
    plt.savefig(noise_data_file[:-3]+'_'+savefig+'_PSD.png')
    # plt.close()

def PSDs_and_cleaning(noise_data_file,VNA_file,char_zs=None,char_fs=None,extra_dec=200,MB_results=None,num_chunks=100,blank_chunks=40,removal_decimation=1,i=0):

    if type(char_zs) == type(None):
        resonator = False
    else:
        resonator = True
        print("Will convert to resonator basis!")

    if type(MB_results) == type(None):
        quasiparticle = False
    else:
        quasiparticle = True
        print("Will convert to quasiparticle basis!")

    timestreams = {}

    ## Pull the raw data from VNA file for this acquisiton
    ## VNA_f is in MHz ; VNA_z is S21 in complex form 
    VNA_f, VNA_z = PUf.read_vna(VNA_file)
    # Plan to modify this for more than two tones eventually
    # print(noise_data_file)

    ## Get the raw noise timestream data and pull some metadata
    data_noise, data_info = PUf.unavg_noi(noise_data_file)
    data_freqs = data_info['search freqs']
    time = data_info['time']
    # print(time[-1])
    time_correction = data_info['sampling period']
    powers = data_info['powers']

    # print(data_freqs)
    num_freqs = len(data_freqs)

    # extra_dec is necessary if significant beating in band
    if extra_dec:
        print('doing additional decimation')
        data_noise = average_decimate(data_noise,extra_dec)
        time_correction *= extra_dec
        time = time[::extra_dec]

    fs = int(1./time_correction)
    print("sampling frequency plugged into welch is " + str(fs))
    

    ## Search the timestream for pulses, break into events, save them to the same file
    pulse_times = storeEvents(noise_data_file,trig_th=4.5,trig_channel='radius')
    print('found ' + str(len(pulse_times)) + ' pulses')

    ## Break data up from a 2D array into an array of 2D arrays, each of a fixed size
    chunk_len = int(len(data_noise) / num_chunks)
    all_chunks = range(num_chunks)
    chunked_timestreams = create_chunks(data_noise,num_chunks)
    chunked_time = create_chunks(time,num_chunks)
    print("chunk length plugged into welch is " + str(chunk_len))
    # print(chunked_time[0:10,0:10,0])

    print('chunked data into '+  str(num_chunks) + ' timestreams')

    ## Check for any chunks that contain a pulse
    bad_chunks = identify_bad_chunks(chunked_time,pulse_times)
    bad_chunks += range(blank_chunks)

    ## Remove the set of bad chunks from the set of all chunks to get good chunks
    good_chunks = list(set(all_chunks)-set(bad_chunks))
    num_good_chunks = len(good_chunks)
    num_bad_chunks = len(bad_chunks)

    ## Now convert the data into the electronics basis: radius & arclength
    radius, arc, _, _ = electronics_basis(data_noise,'multiple freqs')

    ## Converting to absolute radius and arc-length units
    radius_data, arc_data, radius_average, angle_average = electronics_basis(chunked_timestreams,'multiple freqs')
    print('computed electronics basis')

    ## Separate only the good chunks out of the chunked timestream
    timestreams_good_chunks = chunked_timestreams[:,good_chunks,:]
    radius_good_chunks = radius_data[:,good_chunks,:]
    arc_good_chunks = arc_data[:,good_chunks,:]
    time_good_chunks = chunked_time[:,good_chunks,:]

    ## Reshape the timestreams to be flat 2D, with chunks containing pulses removed
    radius_no_pulse = np.reshape(radius_good_chunks,(num_good_chunks*chunk_len,num_freqs),order='F')
    arc_no_pulse = np.reshape(arc_good_chunks,(num_good_chunks*chunk_len,num_freqs),order='F')
    time_no_pulse = np.reshape(time_good_chunks,num_good_chunks*chunk_len,order='F')
    timestreams_no_pulse = np.reshape(timestreams_good_chunks,(num_good_chunks*chunk_len,num_freqs),order='F')

    ## Remove the noise from each of the two timestream directions
    print('cleaning...')
    radius_clean, rad_coeff = noise_removal(radius_no_pulse,removal_decimation=removal_decimation)
    arc_clean, arc_coeff = noise_removal(arc_no_pulse,removal_decimation=removal_decimation)

    # ## Do a simpler cleaning too for some reason?
    # radius_clean_simple, simp_rad_coeff = noise_removal_simple(radius_no_pulse[:,0],radius_no_pulse[:,1],radius_average[0],radius_average[1])
    # arc_clean_simple, simp_arc_coeff = noise_removal_simple(arc_no_pulse[:,0],arc_no_pulse[:,1],radius_average[0],radius_average[1])

    ## Store the timestreams and cleaning coefficients in a dictionary
    timestreams['time'] = time_no_pulse
    timestreams['radius'] = radius_clean
    timestreams['arc'] = arc_clean
    timestreams['radius uncleaned'] = radius_no_pulse
    timestreams['arc uncleaned'] = arc_no_pulse
    # timestreams['radius simple clean'] = radius_clean_simple
    # timestreams['arc simple clean'] = arc_clean_simple
    timestreams['radius coefficient'] = rad_coeff
    timestreams['arc coefficient'] = arc_coeff
    # timestreams['simple radius coefficient'] = simp_rad_coeff
    # timestreams['simple arc coefficient'] = simp_arc_coeff

    # print(radius_clean.shape,arc_clean.shape)

    ## Save the cleaned timestreams to a new file
    timestreams_clean = save_clean_timestreams(noise_data_file,radius_average,angle_average,radius_clean,arc_clean,fs,cd1_coeff=rad_coeff,cd2_coeff=arc_coeff,override=True)
    # timestreams_clean_simple = save_clean_timestreams(noise_data_file,data_noise,radius_clean_simple,arc_clean_simple,fs,override=False)


    print('number of chunks used to average is ' + str(num_good_chunks))

    ## Do a welch signal processing
    f,P_radius_avg       = welch(radius_no_pulse,fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
    f,P_arc_avg          = welch(arc_no_pulse,   fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
    f,P_radius_clean_avg = welch(radius_clean,   fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
    f,P_arc_clean_avg    = welch(arc_clean,      fs=fs,noverlap=0,nperseg=chunk_len,axis=0)

    PSDs_radius = np.zeros(P_radius_clean_avg.shape)
    PSDs_radius[:,0] = P_radius_clean_avg[:,0]
    PSDs_radius[:,1] = P_radius_avg[:,1]

    PSDs_arc = np.zeros(P_arc_clean_avg.shape)
    PSDs_arc[:,0] = P_arc_clean_avg[:,0]
    PSDs_arc[:,1] = P_arc_avg[:,1]

    ## Create and fill a dictionary for the PSD values
    PSDs = {}
    PSDs['f'] = f
    PSDs['radius'] = PSDs_radius
    PSDs['arc'] = PSDs_arc
    PSDs['radius uncleaned'] = P_radius_avg[:,0]
    PSDs['arc uncleaned'] = P_arc_avg[:,0]
    # print(PSDs_radius.shape,timestreams_clean.shape,np.mean(abs(timestreams_clean)).shape)
    PSDs['fractional radius'] = PSDs['radius']/np.mean(abs(timestreams_clean),dtype=np.float64)**2
    PSDs['fractional arc'] = PSDs['arc']/np.mean(abs(timestreams_clean),dtype=np.float64)**2

    ## If we have provided char_fs and char_zs, we can do the resonator transformation
    if resonator:
        print("Converting to resonator basis!")
        frequency, dissipation, ideal, res \
                          = resonator_basis(timestreams_no_pulse[:,0],\
                                                            data_freqs[0],\
                                                            VNA_f,VNA_z,\
                                                            char_fs,char_zs)
        frequency_clean, dissipation_clean, ideal_clean, _ \
                          = resonator_basis(timestreams_clean[:,0],\
                                            data_freqs[0],\
                                            VNA_f,VNA_z,\
                                            char_fs,char_zs)
        # frequency_clean_simple, dissipation_clean_simple, ideal_clean_simple, _ \
        #                   = resonator_basis(timestreams_clean_simple[:,0],\
        #                                     data_freqs[0],\
        #                                     VNA_f,VNA_z,\
        #                                     char_fs,char_zs)

        f,P_frequency = welch(frequency,fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
        f,P_dissipation = welch(dissipation,fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
        f,P_frequency_clean = welch(frequency_clean,fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
        f,P_dissipation_clean = welch(dissipation_clean,fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
        # f,P_frequency_clean_simple = welch(frequency_clean_simple,fs=fs,noverlap=0,nperseg=chunk_len,axis=0)
        # f,P_dissipation_clean_simple = welch(dissipation_clean_simple,fs=fs,noverlap=0,nperseg=chunk_len,axis=0)

        PSDs['dissipation uncleaned'] = P_dissipation
        PSDs['frequency uncleaned'] = P_frequency

        PSDs['dissipation'] = P_dissipation_clean
        PSDs['frequency'] = P_frequency_clean

        timestreams['dissipation'] = dissipation_clean
        timestreams['frequency'] = frequency_clean

        # PSDs['dissipation simple clean'] = P_dissipation_clean_simple
        # PSDs['frequency simple clean'] = P_frequency_clean_simple
        #
        # timestreams['dissipation simple clean'] = dissipation_clean_simple
        # timestreams['frequency simple clean'] = frequency_clean_simple

        ## If we have provided MB fit results, we can do the quasiparticle transformation
        if quasiparticle:
            print("Converting to quasiparticle basis!")
            P_k1_clean, P_k2_clean = quasiparticle_basis(data_T=0.011,\
                                                         dissipation=np.sqrt(P_dissipation_clean),\
                                                         frequency=np.sqrt(P_frequency_clean),\
                                                         MB_results=MB_results,\
                                                         readout_f=data_freqs[0])
            P_k1, P_k2 = quasiparticle_basis(data_T=0.011,\
                                                         dissipation=np.sqrt(P_dissipation),\
                                                         frequency=np.sqrt(P_frequency),\
                                                         MB_results=MB_results,\
                                                         readout_f=data_freqs[0])


            PSDs['kappa_1'] = P_k1_clean**2
            PSDs['kappa_2'] = P_k2_clean**2

            PSDs['kappa_1 uncleaned'] = P_k1**2
            PSDs['kappa_2 uncleaned'] = P_k2**2

            k1_timestream_clean, k2_timestream_clean = quasiparticle_basis(data_T=0.011,\
                                                         dissipation=dissipation_clean,\
                                                         frequency=frequency_clean,\
                                                         MB_results=MB_results,\
                                                         readout_f=data_freqs[0])

            # k1_timestream_clean_simple, k2_timestream_clean_simple = quasiparticle_basis(data_T=0.011,\
            #                                              dissipation=dissipation_clean_simple,\
            #                                              frequency=frequency_clean_simple,\
            #                                              MB_results=MB_results,\
            #                                              readout_f=data_freqs[0])


            timestreams['k1'] = k1_timestream_clean
            timestreams['k2'] = k2_timestream_clean
            # timestreams['k1 simple clean'] = k1_timestream_clean_simple
            # timestreams['k2 simple clean'] = k2_timestream_clean_simple

    if False:

        # print(radius_data.shape,arc_data.shape)
        # Calculating PSDs before and after cleaning by iterating
        # over chunks
        # num_good_chunks = 0
        # PSD_length = int(len(chunked_timestreams)/2+1)
        # P_radius_sum = np.zeros((PSD_length,num_freqs))
        # P_arc_sum = np.zeros((PSD_length,num_freqs))
        # P_radius_clean_sum = np.zeros((PSD_length,num_freqs))
        # P_arc_clean_sum = np.zeros((PSD_length,num_freqs))
        # chunk_time = time[-1]/num_chunks
        # radius_no_pulse = np.array([[],[]])
        # arc_no_pulse = np.array([[],[]])
        # time_no_pulse = np.array([[],[]])
        # f = fft.fftfreq(int(len(data_noise)/num_chunks),d=time_correction)
        # iter = 0
        # for chunk in good_chunks:
        #
        #     radius_chunk = np.squeeze(radius_data[:,chunk,:])
        #     arc_chunk = np.squeeze(arc_data[:,chunk,:])
        #     time_chunk = np.squeeze(chunked_time[:,chunk,:])
        #
        #     if iter == 0:
        #         radius_no_pulse = radius_chunk
        #         arc_no_pulse = arc_chunk
        #         time_no_pulse = time_chunk
        #     else:
        #         radius_no_pulse = np.concatenate((radius_no_pulse,radius_chunk),axis=0)
        #         arc_no_pulse = np.concatenate((arc_no_pulse,arc_chunk),axis=0)
        #         time_no_pulse = np.concatenate((time_no_pulse,time),axis=0)
        #     iter += 1
            # print(radius_chunk.shape,arc_chunk.shape)


            # f, P_radius = periodogram(radius_chunk,fs=fs,axis=0)
            # _, P_arc = periodogram(arc_chunk,fs=fs,axis=0)

            # radius_chunk_clean = noise_removal(radius_chunk,removal_decimation=50)
            # arc_chunk_clean = noise_removal(arc_chunk,removal_decimation=50)
            #
            # _, P_radius_clean = periodogram(radius_chunk_clean,fs=fs,axis=0)
            # _, P_arc_clean = periodogram(arc_chunk_clean,fs=fs,axis=0)
            #
            # radius_clean = np.concatenate(radius_clean,radius_chunk_clean,axis=1)
            # arc_clean = np.concatenate(arc_clean,arc_chunk_clean,axis=1)
            # time_clean = np.concatenate(time_clean,time_chunk,axis=1)

            # P_radius_sum += CSD(chunk_time,radius_chunk,radius_chunk).real[f>=0]
            # P_arc_sum += CSD(chunk_time,arc_chunk,arc_chunk).real[f>=0]
            #
            # P_radius_clean_sum += CSD(chunk_time,radius_chunk_clean,radius_chunk_clean).real[f>=0]
            # P_arc_clean_sum += CSD(chunk_time,arc_chunk_clean,arc_chunk_clean).real[f>=0]

            # P_radius_sum += P_radius
            # P_arc_sum += P_arc

            # P_radius_clean_sum += P_radius_clean
            # P_arc_clean_sum += P_arc_clean

            # num_good_chunks += 1
            # if chunk % 10 == 0:
            #     print('finished chunk ' + str(chunk))


        # with h5py.File(noise_data_file, 'r+') as fyle:
        #     fyle["cleaned_data"] =


        # P_radius_avg = P_radius_sum / num_good_chunks
        # P_arc_avg = P_arc_sum / num_good_chunks

        # P_radius_clean_avg = P_radius_clean_sum / num_good_chunks
        # P_arc_clean_avg = P_arc_clean_sum / num_good_chunks


        # And a bunch of plots
        filler = False

    timestream_plot = True
    if timestream_plot:
        radius_std = np.std(radius_no_pulse[:,0])
        num_freqs = int(len(data_freqs))
        visual_separation = np.linspace(0,20*num_freqs*radius_std,num_freqs)
        

        plt.figure(noise_data_file + '_raw_S21')
        plt.plot(time_no_pulse,radius_no_pulse[:,0],color='C0')
        for f_idx in range(1,num_freqs):
            plt.plot(time,radius[:,f_idx] + visual_separation[f_idx],color=c_wheel_0[f_idx])
        plt.plot(chunked_time[:,bad_chunks,0],\
                 radius_data[:,bad_chunks,0],color='r')
        plt.title('radius timestream')
        plt.ylabel('ADC units')
        plt.xlabel('time (s)')
        

        plt.figure(noise_data_file + '_timestream')
        plt.plot(time_no_pulse,arc_no_pulse[:,0],color='C0')
        for f_idx in range(1,num_freqs):
            plt.plot(time,arc[:,f_idx] + visual_separation[f_idx],color=c_wheel_0[f_idx])
        plt.plot(chunked_time[:,bad_chunks,0],\
                 arc_data[:,bad_chunks,0],color='r')
        plt.title('arc length timestream')
        plt.ylabel('ADC units')
        plt.xlabel('time (s)')

        VNA_f, VNA_z = PUf.read_vna(VNA_file)
        f_idx = PUf.find_closest(VNA_f,data_freqs[0])
        # print VNA_f, data_freqs[0], f_idx
        # PUf.plot_noise_and_vna(timestreams_no_pulse,VNA_z,\
        #                        fit_z=None,char_zs=char_zs,alpha=0.05,title='off resonance timestream -- raw S21')
        PUf.plot_noise_and_vna(timestreams_clean[:,0],VNA_z,\
                               fit_z=None,char_zs=char_zs,alpha=0.05,title=str(i))
        # plt.savefig(noise_data_file[:-3]+'_S21.png')
        # timestreams_bad_chunks = chunked_timestreams[:,bad_chunks,:]
        # timestream_bad_chunks = np.reshape(timestreams_bad_chunks,(num_bad_chunks*chunk_len,num_freqs),order='F')
        # PUf.plot_noise_and_vna(timestream_bad_chunks,VNA_z,f_idx=f_idx,char_zs=char_zs,alpha=1)
        # plt.show()
        # plt.show(False)
        # plt.pause(5)
        # raw_input('press enter to close all plots')
        # plt.close('all')

    resonator_timestream_plot = True
    if resonator_timestream_plot and resonator:
        # plt.plot(time_no_pulse,frequency)
        # plt.plot(time_no_pulse,frequency_clean+10*np.std(frequency))
        # plt.title('df/f timestream')
        # plt.figure()
        # plt.plot(time_no_pulse,dissipation)
        # plt.plot(time_no_pulse,dissipation_clean+10*np.std(dissipation))
        #
        # plt.title('d(1/Q) timestream')

        f_idx = PUf.find_closest(ideal_clean['f'],data_freqs[0])

        PUf.plot_noise_and_vna(ideal_clean['timestream'],ideal_clean['z'],\
                               char_zs=ideal_clean['char z'],title=noise_data_file+'_ideal')#,i=i)
        # plt.show()
        # plt.show(False)
        # plt.pause(5)
        # raw_input('press enter to close all plots')
        # plt.close('all')

    PSD_plotting = True
    if PSD_plotting:
        # print(P_radius_avg.shape,P_arc_avg.shape)
        plot_PSDs(f,P_radius_avg,P_arc_avg,\
                  noise_data_file,directions=['radius','arc length'],\
                  units=['ADCu','ADCu'],savefig='electronics',data_freqs=data_freqs,\
                  title='',P_1_clean=P_radius_clean_avg,P_2_clean = P_arc_clean_avg)
        # plt.show()
        # plt.show(False)
        # plt.pause(5)
        # raw_input('press enter to close all plots')
        # plt.close('all')
        if resonator:
            plot_PSDs(f,P_dissipation,P_frequency,\
                  noise_data_file,directions=['dissipation','frequency'],\
                  units=['d(1/Q)','df/f'],savefig='resonator',data_freqs=[data_freqs[0]],\
                  title='',P_1_clean=P_dissipation_clean,P_2_clean=P_frequency_clean)

            if quasiparticle:
                plot_PSDs(f,P_k1,P_k2,\
                  noise_data_file,directions=['kappa1','kappa2'],\
                  units=['um^-3','um^-3'],savefig='nqp',data_freqs=[data_freqs[0]],\
                  title='',P_1_clean=P_k1_clean,P_2_clean=P_k2_clean)

    if resonator:
        return powers, PSDs, res, timestreams
    else:
        return powers, PSDs, None, timestreams

def coherence_analysis(noise_data_file,extra_dec=None):
    # Plan to modify this for more than two tones eventually
    data_freqs, data_noise, time_correction,_ = PUf.unavg_noi(noise_data_file)

    # extra_dec is necessary if significant beating in band
    if extra_dec:
        print('doing additional decimation')
        resonance_dec = sig.decimate(data_noise[:,0],extra_dec)
        tracking_dec = sig.decimate(data_noise[:,1],extra_dec)
        data_noise = np.array([resonance_dec,tracking_dec]).T
        time_correction *= extra_dec

    # Checks if the beating is too large (phase will appear as shark tooth)
    clipping = any(abs(np.mean(data_noise,axis=0)) <= np.mean(abs(data_noise-np.mean(data_noise,axis=0)),axis=0))

    # Converting to absolute and arc-length units
    # coh_data_1 = abs(data_noise)
    # coh_data_2 = np.angle(data_noise*np.exp(-1j*np.angle(np.mean(data_noise,axis=0))))*np.mean(abs(data_noise),axis=0)

    coh_data_1,coh_data_2 = electronics_basis(data_noise)

    # Calculating cross-PSDs (for coherence and PSD comparisons)
    J_freqs, CSD_avg_1 = CSD_avg(coh_data_1, time_correction, 50)
    J_freqs, CSD_avg_2 = CSD_avg(coh_data_2, time_correction, 50)

    # Cleaning won't work if phase is a shark tooth
    clipping = 1
    if clipping == 0:
        # Decimate, fit noise time streams, clean undecimated data
        coh_data_1_clean = noise_removal(coh_data_1,removal_decimation=500)
        coh_data_2_clean = noise_removal(coh_data_2,removal_decimation=500)

        # Clean cross-PSDs
        J_freqs, CSD_avg_1_clean = CSD_avg(coh_data_1_clean, time_correction, 50)
        J_freqs, CSD_avg_2_clean = CSD_avg(coh_data_2_clean, time_correction, 50)

        # Save cleaned data back to original file
        save_clean_timestreams(noise_data_file,data_noise,coh_data_1_clean,coh_data_2_clean,override=True)

    # And a bunch of plots
    fig_0, axes_0 = plt.subplots(2,len(data_freqs)+1,sharex=True,sharey='row',figsize=(40,9))
    for Nm in range(len(data_freqs)):
        if Nm == 0:
            axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1[Nm,Nm][J_freqs>0].real,c=c_wheel_0[Nm],label='on-res decimated') # ComplexWarning: Casting complex values to real discards the imaginary part
            axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_0[Nm],label='on-res decimated')
            if clipping == 0:
                axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1_clean[Nm,Nm][J_freqs>0].real,c=c_wheel_1[Nm],label='on-res cleaned')
                axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_1[Nm],label='on-res cleaned')
        elif Nm == 1:
            axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1[Nm,Nm][J_freqs>0].real,c=c_wheel_0[Nm],label='tracking decimated') # ComplexWarning: Casting complex values to real discards the imaginary part
            axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_0[Nm],label='tracking decimated')
            if clipping == 0:
                axes_0[0,0].loglog(J_freqs[J_freqs>0],CSD_avg_1_clean[Nm,Nm][J_freqs>0].real,c=c_wheel_1[Nm],label='tracking cleaned')
                axes_0[1,0].loglog(J_freqs[J_freqs>0],CSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_1[Nm],label='tracking cleaned')
        axes_0[0,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_1[Nm,Nm][J_freqs>0].real,c=c_wheel_0[Nm],label=str(round(data_freqs[Nm],3))+' decimated')
        axes_0[1,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_2[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_0[Nm],label=str(round(data_freqs[Nm],3))+' decimated')
        if clipping == 0:
            axes_0[0,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_1_clean[Nm,Nm][J_freqs>0].real,c=c_wheel_1[Nm],label=str(round(data_freqs[Nm],3))+' cleaned')
            axes_0[1,Nm+1].loglog(J_freqs[J_freqs>0],CSD_avg_2_clean[Nm,Nm][J_freqs>0].real,'--',c=c_wheel_1[Nm],label=str(round(data_freqs[Nm],3))+' cleaned')
        axes_0[0,Nm+1].legend()
        axes_0[0,Nm+1].tick_params(labelbottom=True,labelleft=True)
        axes_0[1,Nm+1].legend()
        axes_0[1,Nm+1].tick_params(labelbottom=True,labelleft=True)
    axes_0[0,0].legend()
    axes_0[0,0].tick_params(labelbottom=True,labelleft=True)
    axes_0[1,0].legend()
    axes_0[1,0].tick_params(labelbottom=True,labelleft=True)

    fig_0.add_subplot(211, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.ylabel('absolute value PSD [(ADCu)$^2 Hz^{-1}$]',fontsize=12,labelpad=20)

    fig_0.add_subplot(212, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.ylabel('arc length PSD [(ADCu)$^2 Hz^{-1}$]',fontsize=12,labelpad=20)
    plt.xlabel('Frequency [Hz]',fontsize=25,labelpad=20)

    plt.savefig(noise_data_file[:-3]+'_0_PSD.png')
    plt.close()

    fig_1, axes_1 = plt.subplots(len(data_freqs),len(data_freqs),sharex=True,sharey=True,figsize=(30,18))
    average_all_dec = 0
    #average_all_clean = 0
    for Nm in range(len(data_freqs)):
        average_Nm_dec = np.mean(np.delete((abs(CSD_avg_1[:,Nm])**2)/(np.diagonal(CSD_avg_1,axis1=0,axis2=1).T.real*CSD_avg_1[Nm,Nm].real),Nm,axis=0),axis=0)
        average_all_dec += (1/len(data_freqs))*average_Nm_dec
        axes_1[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_dec[J_freqs>0],c=c_wheel_0[Nm])
        #if clipping == 0:
        #    average_Nm_clean = np.mean(np.delete((abs(CSD_avg_1_clean[:,Nm])**2)/(np.diagonal(CSD_avg_1_clean,axis1=0,axis2=1).T.real*CSD_avg_1_clean[Nm,Nm].real),Nm,axis=0),axis=0)
        #    average_all_clean += (1/len(data_freqs))*average_Nm_clean
        #    axes_1[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_clean[J_freqs>0],'--',c=c_wheel_1[Nm])
        axes_1[Nm,Nm].tick_params(labelbottom=True,labelleft=True)
        for Nn in range(len(data_freqs)):
            if Nm < Nn:
                axes_1[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_1[Nn,Nm][J_freqs>0])**2)/(CSD_avg_1[Nn,Nn][J_freqs>0].real*CSD_avg_1[Nm,Nm][J_freqs>0].real),c='gray')
                #if clipping == 0:
                #    axes_1[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_1_clean[Nn,Nm][J_freqs>0])**2)/(CSD_avg_1_clean[Nn,Nn][J_freqs>0].real*CSD_avg_1_clean[Nm,Nm][J_freqs>0].real),'--',c='gray')
                axes_1[Nn,Nm].legend(handles=[mpatches.Patch(color=c_wheel_0[Nn]),mpatches.Patch(color=c_wheel_0[Nm])])
                axes_1[Nn,Nm].tick_params(labelbottom=True,labelleft=True)
            elif Nm != Nn:
                axes_1[Nn,Nm].axis('off')
    axes_1[0,len(data_freqs)-1].axis('on')
    axes_1[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_dec[J_freqs>0],c='k')
    #if clipping == 0:
    #    axes_1[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_clean[J_freqs>0],'--',c='k')
    axes_1[0,len(data_freqs)-1].tick_params(labelbottom=True,labelleft=True)

    fig_1.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.xlabel('Frequency [Hz]',fontsize=25,labelpad=20)
    plt.ylabel('absolute value coherence',fontsize=25,labelpad=20)

    plt.savefig(noise_data_file[:-3]+'_0_coh_a.png')
    plt.close()

    fig_2, axes_2 = plt.subplots(len(data_freqs),len(data_freqs),sharex=True,sharey=True,figsize=(30,18))
    average_all_dec = 0
    #average_all_clean = 0
    for Nm in range(len(data_freqs)):
        average_Nm_dec = np.mean(np.delete((abs(CSD_avg_2[:,Nm])**2)/(np.diagonal(CSD_avg_2,axis1=0,axis2=1).T.real*CSD_avg_2[Nm,Nm].real),Nm,axis=0),axis=0)
        average_all_dec += (1/len(data_freqs))*average_Nm_dec
        axes_2[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_dec[J_freqs>0],c=c_wheel_0[Nm])
        #if clipping == 0:
        #    average_Nm_clean = np.mean(np.delete((abs(CSD_avg_2_clean[:,Nm])**2)/(np.diagonal(CSD_avg_2_clean,axis1=0,axis2=1).T.real*CSD_avg_2_clean[Nm,Nm].real),Nm,axis=0),axis=0)
        #    average_all_clean += (1/len(data_freqs))*average_Nm_clean
        #    axes_2[Nm,Nm].semilogx(J_freqs[J_freqs>0],average_Nm_clean[J_freqs>0],'--',c=c_wheel_1[Nm])
        axes_2[Nm,Nm].tick_params(labelbottom=True,labelleft=True)
        for Nn in range(len(data_freqs)):
            if Nm < Nn:
                axes_2[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_2[Nn,Nm][J_freqs>0])**2)/(CSD_avg_2[Nn,Nn][J_freqs>0].real*CSD_avg_2[Nm,Nm][J_freqs>0].real),c='gray')
                #if clipping == 0:
                #    axes_2[Nn,Nm].semilogx(J_freqs[J_freqs>0],(abs(CSD_avg_2_clean[Nn,Nm][J_freqs>0])**2)/(CSD_avg_2_clean[Nn,Nn][J_freqs>0].real*CSD_avg_2_clean[Nm,Nm][J_freqs>0].real),'--',c='gray')
                axes_2[Nn,Nm].legend(handles=[mpatches.Patch(color=c_wheel_0[Nn]),mpatches.Patch(color=c_wheel_0[Nm])])
                axes_2[Nn,Nm].tick_params(labelbottom=True,labelleft=True)
            elif Nm != Nn:
                axes_2[Nn,Nm].axis('off')
    axes_2[0,len(data_freqs)-1].axis('on')
    axes_2[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_dec[J_freqs>0],c='k')
    #if clipping == 0:
    #    axes_2[0,len(data_freqs)-1].semilogx(J_freqs[J_freqs>0],average_all_clean[J_freqs>0],'--',c='k')
    axes_2[0,len(data_freqs)-1].tick_params(labelbottom=True,labelleft=True)

    fig_2.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    #plt.title(str(readout_power[readout])+' dBm Readout Power',fontsize=20)
    plt.xlabel('Frequency [Hz]',fontsize=25,labelpad=20)
    plt.ylabel('arc length coherence',fontsize=25,labelpad=20)

    plt.savefig(noise_data_file[:-3]+'_0_coh_b.png')
    plt.close()

def readDataFile(filestr):
    data, data_info = PUf.unavg_noi(filestr)
    sampling_period = data_info['sampling period']
    times = data_info['time']

    sampling_rate = 1./sampling_period
    res = dict()
    radius, arc_length,_,_ = electronics_basis(data[:,0])
    res['radius']=radius
    res['arc length'] = arc_length
    res['Time']=times
    res['Fs']=sampling_rate
    res['number_samples']=len(data)
    res['chan_names']=['radius','arc length']
    return res

def pulse(x,x0,tau1=20e-6,tau2=200e-6):
    dx=(x-x0)
    dx*=np.heaviside(dx,1)
    return (np.exp(-dx/tau1)-np.exp(-dx/tau2))/(tau1-tau2)*np.heaviside(dx,1)

def pulseFromTemplate(template,noisepsd,fs):

    # calculate the time-domain optimum filter
    phi = irfft(rfft(template)/noisepsd).real
    # calculate the normalization of the optimum filter
    norm = np.dot(phi, template)

    # calculate the expected energy resolution
    resolution = 1.0/(np.dot(phi, template)/fs)**0.5

    template = phi/norm*fs

    return template,[phi,norm,resolution]

## Searches a given channel in a given file for pulses. Saves a list of pulse times
## back to that same file, overwriting what's there if desired
def storeEvents(filename,override=True,trig_th=4,trig_channel='arc length'):
    res = getEvents(filename,trig_th=trig_th,trig_channel=trig_channel)
    pulse_times = res['trigtime']
    with h5py.File(filename, 'r+') as fyle:
        if 'pulses' in fyle.keys():
            print('pulse time data already exists! If you set override=False, nothing will happen.')
            if override:
                print('saving pulse time data to {} because override=True!'.format(filename))
                del fyle['pulses']
                fyle['pulses'] = pulse_times
        else:
            print('saving pulses to {}!'.format(filename))
            fyle['pulses'] = pulse_times
    return pulse_times

def getEvents(filename, trig_channel='arc length', trig_th = 4.0,\
              rising_edge = True, maxAlign=True, pretrig = 2000,\
              trace_len=10000,trig_sep=10000,ds=1,\
              pretrig_template = 2000, tauRise=5e-6,tauFall=50e-6,\
              ACcoupled=True,verbose=False,template=None):

    """
    finds the pulse events in a timestream on either the arc length or radius channel

    input: noise timestream file
    output: res dictionary which will have the times of the pulses


    Below from Noah Kurinsky:
    This function takes data from continuous DAQ and slices the data into events with a level trigger

    Double exponential template is generated if no template is given. For OF template, run 'pulseFromTemplate' to generate
    apropriate template for use with trigger function
    """

    res=readDataFile(filename)
    chan_names=res['chan_names']
    number_samples=res['number_samples']

    #make sure trigger channel is valid
    chan_names = res['chan_names']
    if(trig_channel not in chan_names):
        trig_channel='arc length'
        if(verbose):
            print('Trigger Channel Defaulting to Phase')

    #setup trigger template
    fs=res['Fs']
    dt=1.0/fs
    if(template is None):
        #produce shaping template
        pretrigger=pretrig_template*dt
        xtemplate=np.arange(0,trace_len)*dt
        template=pulse(xtemplate,pretrigger,tau1=tauRise,tau2=tauFall)
    else:
        oldtl = trace_len
        trace_len = len(template)
        if(oldtl == trig_sep):
            trig_sep = trace_len
        pretrig_template = np.argmax(template)
        xtemplate=np.arange(0,trace_len)*dt

    if(ACcoupled): #removes DC component; flat trace gives 0
        template-=np.mean(template)

    trace=res[trig_channel]
    #downsample and average template and trace
    meandt=dt*ds
    trig_sep_ds = int(trig_sep/ds)

    meanTemplate=np.mean(template.reshape(int(len(template)/ds),ds),axis=1)
    meanTrace=np.mean(trace.reshape(int(len(trace)/ds),ds),axis=1)



    #pulse shaping maintainin correct amplitude
    filtered_data = scipy.signal.fftconvolve(meanTrace, meanTemplate[::-1], mode="valid")

    convolution_mean = np.mean(filtered_data)
    trigger_std = trig_th*np.std(filtered_data)
    # filtered_data = np.correlate(meanTrace,meanTemplate)*meandt
    # plt.plot(res['radius'])
    # plt.figure()
    # plt.plot(filtered_data)
    # plt.show()

    if (rising_edge): #rising edge
        if(verbose):
            print('Triggering on rising edge')
        trigA = (filtered_data[0:-1] < convolution_mean + trigger_std)
        trigB = (filtered_data[1:] > convolution_mean + trigger_std)
    else: #falling edge
        if(verbose):
            print('Triggering on falling edge')
        trigA = (filtered_data[0:1] > trigger_std)
        trigB = (filtered_data[1:] < trigger_std)
    trigger_condition = trigA & trigB
    trigger_points=np.flatnonzero(trigger_condition)+1

    rm_index = []
    n_trig = len(trigger_points)
    n_trig_pts = n_trig
    idx = 0
    alignPreTrig = 200
    alignPostTrig = 500
    while (idx < n_trig-2):

        #remove redundant triggers
        nidx = idx + 1
        while ( (nidx< n_trig) and ((trigger_points[nidx] - trigger_points[idx])< trig_sep_ds) ):
            rm_index.append(nidx)
            nidx += 1

        #update loop
        idx = nidx

    if(len(rm_index) > 0):
        rm_index = np.array(rm_index)
        trigger_points = np.delete(trigger_points, rm_index)

    #align trigger with pulse maximum
    if(maxAlign):
        for idx in range(0,len(trigger_points)):
            trigWindowStart=trigger_points[idx] - int(alignPreTrig/ds)
            trigWindowEnd=trigger_points[idx] + int(alignPostTrig/ds)
            if(trigWindowStart > 0 and trigWindowEnd < len(filtered_data)):
                trigger_points[idx] = np.argmax(filtered_data[trigWindowStart:trigWindowEnd])+trigWindowStart

    if(ds > 1):
        trigger_points*=ds
    trigger_points += pretrig_template
    n_trig = len(trigger_points)

    for ch_str in chan_names:
        rm_index = []
        singleTrace=res[ch_str]
        res[ch_str] = []
        for i in range(0,len(trigger_points)):
            trigpt = trigger_points[i]

            #avoid traces too close to the edge of the trace
            trigAreaStart=trigpt - pretrig
            trigAreaEnd=trigpt + trace_len - pretrig
            if(trigAreaStart < 0 or trigAreaEnd > len(singleTrace)):
                rm_index.append(i)
                continue
            res[ch_str].append(singleTrace[trigAreaStart:trigAreaEnd])
        if(len(rm_index) > 0):
            rm_index = np.array(rm_index)
            trigger_points = np.delete(trigger_points, rm_index)
        res[ch_str] = np.array(res[ch_str])

    res['trigpt'] = trigger_points
    res['trigtime'] = res['Time'][trigger_points]
    res['filename'] = np.full(n_trig,filename)
    res['trigRate'] = np.full(n_trig,n_trig)
    res['trigPts'] = np.full(n_trig,n_trig_pts)

    del singleTrace
    del filtered_data
    del trigger_condition
    del trigA,trigB

    return res
