import os, h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import expit

import PyMKID_USRP_functions as PUf
import PyMKID_resolution_functions as Prf
import TimestreamHelperFunctions as Thf

RQ_names = [
        "pre_trig_bl_mean",
        "pre_trig_bl_sdev",
        "post_pls_bl_mean",
        "post_pls_bl_sdev",
        "full_win_max",
        "full_win_maxsamp",
        "pre_pls_max",
        "post_pls_max",
        "peak_pls_max",
]

## This method pulls the metadata from the summary file and extracts the LED pulse
## profile as well as some derived quantities, assuming the LED settings are the 
## same for every file in the dataset (which is normally the case)
## ARGUMENTS
##	- summary_file		<string>			Full path to the h5 file containig run summary information
##	- blank_fraction 	<float>				Fraction of the waveform to ignore at the beginning
## 	- verbose			<bool>				Flag to show text ouptut
## RETURNS
##	- voltages			<array of float>	Ordered numpy array with LED voltage settings in V
##	- p_params			<dictionary>		All explicit and derived quantities of pulse profile
##	- charFs			<array of float>	Array of arrays (# noise x # tones) - readout F in Hz
##	- charZs			<array of complex>	Array of arrays (# noise x # tones) - mean S21
def parse_metadata(summary_file, blank_fraction=0.2, verbose=False):

    ## Load the summary file
    ## Show this function's output only if verbosity level 2 or greater selected
    md, charFs, charZs = Thf.UnpackSummary(summary_file, verbose=True if verbose > 1 else False)
    rf_power = md['power'] + md['tx_gain']
    
    if verbose > 0:
        print("RF Power at USRP:", rf_power, "dBm")
        print("Char. freqs: ", charFs)
        print("Char. S21s:  ", charZs)

    ## Extract the voltages used in this run
    voltages = np.array([])

    for k in md.keys():
        if 'LaserScan_' in k:
            voltages = np.append(voltages, float(int(1000*md[k]['LEDvoltage'][0]))/1000.)

    n_volts  = len(voltages)

    ## Extract the pulse profile settings, assuming they're all the same
    for k in md.keys():
        if 'LaserScan_' in k:

            ## Pull the total duration of the aqcuisition if it's in the file, otherwise use the default
            if 'duration' in md[k].keys():
                lapse_sec   = md[k]['duration'][0]
            else:
                lapse_sec   = 10.0

            ## Pull the pulse rate if it's in the file, otherwise use the default
            if 'LEDfreqHz' in md[k].keys():
                LED_rate_Hz = md[k]['LEDfreqHz'][0]
            else:
                LED_rate_Hz = 80.0

            ## Pull the pulse width if it's in the file, otherwise use the default
            if 'LEDpulseus' in md[k].keys():
                pulse_w     = md[k]['LEDpulseus'][0]
            else:
                pulse_w     = 1.0

            ## Pull the pulse delay if it's in the file, otherwise use the default
            if 'delayms' in md[k].keys():
                delay_ms    = md[k]['delayms'][0]
            else: 
                delay_ms    = 5.0

            break ## Stop the loop after finding the first laser scan file

    if verbose > 0:
        print("Duration:", lapse_sec  , "sec")
        print("Pulse f: ", LED_rate_Hz, "Hz")
        print("Pls wdth:", pulse_w    , "us")
        print("P  delay:", delay_ms   , "ms")

    ## Calculate some derived timing parameters
    total_pulses   = LED_rate_Hz*lapse_sec
    time_btw_pulse = 1./LED_rate_Hz
    num_pulses     = int(total_pulses * (1 - blank_fraction))

    if verbose > 0:
        print("Total pulse windows in acq.: ",total_pulses)
        print("Time between pulse arrival:  ",time_btw_pulse,"sec")
        print("Number of windows to look at:",num_pulses)

    ## Define a dictionary for the pulse parameters
    p_params = {
        "rf_power"      : rf_power,
        "n_volts"       : n_volts,
        "lapse_sec"     : lapse_sec,
        "LED_rate_Hz"   : LED_rate_Hz,
        "pulse_w"       : pulse_w,
        "delay_ms"      : delay_ms,
        "total_pulses"  : total_pulses,
        "time_btw_pulse": time_btw_pulse,
        "num_pulses"    : num_pulses,
        "blank_fraction": blank_fraction,
    }

    ## Clean up and return
    del md, rf_power, n_volts, lapse_sec, LED_rate_Hz, pulse_w, delay_ms, total_pulses, time_btw_pulse, num_pulses, blank_fraction,
    return voltages, p_params, charFs, charZs

## This method creates some timing parameters that are used in further pulse analysis
## ARGUMENTS
## 	- pulse_file		<string>			Full path to the h5 file containig pulse data
##	- p_params			<dictionary>		Pulse parameter dictionary returned by parse_metadata
##	- decimate_down_to	<float>				Maximum frequency desired in PSDs (in Hz)
##	- pulse_cln_dec		<int>				An explicit decimation factor, used instead of decimate_down_to
## RETURNS
##	- pulse_noise		<array of complex>	Decimated timestream
##	- N					<int>				Total number of samples in each pulse window
##	- T 				<float> 			Total time for each pulse window (in sec)
##	- t 				<array of float> 	Time domain space array for waveforms (in sec)
##	- f 				<array of float>	Frequency space array for PSDs (in Hz)
##	- pulse_fs 			<float> 			Effective sampling rate after decimation (in Hz)
def get_decimated_timestream(pulse_file, p_params, decimate_down_to, pulse_cln_dec=None):
    ## Determine how much additional decimation to apply
    pulse_noise, pulse_info = PUf.unavg_noi(pulse_file)
    pulse_fs = 1./pulse_info['sampling period']
    if (pulse_cln_dec is None):
        pls_cln_dec = int(pulse_fs/decimate_down_to)
    else:
        pls_cln_dec = int(pulse_cln_dec)

    ## Get the decimated frequency step
    pulse_noise = Prf.average_decimate(pulse_noise,pls_cln_dec)
    pulse_fs   /= pls_cln_dec ## sampling_rate
    
    ## Determine the number of samples in, and length of time of, a full pulse window
    N = int(p_params["time_btw_pulse"]*pulse_fs)    ## Total # of samples pulse window
    if (N % 2 == 1):
        N -= 1
    T = N/pulse_fs                        ## Total time of full pulse window

    ## Get time- and frequency- space arrays for the full pulse window
    t,f = Prf.build_t_and_f(N,pulse_fs)

    return pulse_noise, pulse_info, N, T, t, f, pulse_fs

## Create a plot for each pulse arrival window in the timestream and define a pre-trigger 
## region in which we collect statistics on which to do cuts to remove bad windows
## ARGUMENTS
##  - pulse_file        <array of string>   Each entry is full path to the h5 file containig pulse data
##  - noise_file        <string>            Full path to the h5 file containig noise data
##  - vna_file          <string>            Full path to the h5 file containig VNA data
##  - p_params          <dictionary>        Pulse parameter dictionary returned by parse_metadata
##  - p1                <float>             Percentile to draw low-end cut line on distribution plots
##  - p2                <float>             Percentile to draw high-end cut line on distribution plots
##  - decimate_down_to  <float>             Maximum frequency desired in PSDs (in Hz)
##  - pulse_cln_dec     <int>               An explicit decimation factor, used instead of decimate_down_to
##  - PHASE             <bool>              Whether to plot timestreams in phase or log-mag
##  - show_plots        <bool>              Flag to render plots
## RETURNS
##  - bl_means          <array of float>    Array containig the pre-trig baseline mean for each pulse window
##  - bl_sdevs          <array of float>    Array containig the pre-trig baseline sdev for each pulse window
##  - pls_maxs          <array of float>    Array containig the maximum value for each pulse window
## - rqs                        dict containing two dicts (one for each quadrature), each containing lists 
##                              of RQ values, one key per RQ defined above
def plot_pulse_windows(pulse_file, noise_file, vna_file, p_params, bad_pls_idx_arr=[None], pre_trig_sep_ms=0.250, post_pls_sep_ms=2.500, p1=5, p2=90, decimate_down_to=5e4, pulse_cln_dec=None, PHASE=True, show_plots=False,):

    ## Define the time that separates "pre-trigger" region from rest of pulse window
    pretrig_seconds = (p_params["delay_ms"]-0.25)*1e-3

    print('===================')
    print('plotting pulse file:',pulse_file)
    print('using VNA file:     ',vna_file)
    print('using noise file:   ',noise_file)

    ## Get the decimated timestream and frequency step
    pulse_noise, _, N, T, t, f, samp_rate = get_decimated_timestream(pulse_file, p_params, decimate_down_to, pulse_cln_dec)

    ## Define the regions where pulses exist
    ## =====================================
    pretrig_seconds = (p_params['delay_ms']-pre_trig_sep_ms)*1e-3
    postpls_seconds = (p_params['delay_ms']+post_pls_sep_ms)*1e-3
    
    ## This defines where (in # of pulse windows) to start looking for pulse windows
    pulse_start = int(p_params['total_pulses'] * p_params['blank_fraction'])
    samples_per_pulse = int(p_params['time_btw_pulse']*samp_rate)
    
    ## Define some times of interest in units of samples
    pretrig = int(samp_rate * pretrig_seconds) ## Region before pulse rising edge / trigger
    pstpuls = int(samp_rate * postpls_seconds) ## Region after pulse has returned to baseline
    plstrig = int(samp_rate * p_params['delay_ms']*1e-3) ## Sample at the trigger time

    ## Create an output dictionary containing arrays of 2-tuples
    ## Each 2-tuple is a calculated RQ for each event, with the first component as logmag, second as phase
    qdrtrs = ['phase', 'logmag']
    rqs = { q : {key: [] for key in RQ_names} for q in qdrtrs }
    
    ## Create an empty array to store our results in
    ## This is the average baseline of the three timestreams across all pulse windows
    noise_averages = 0 # np.zeros((3),dtype=np.complex128)
    
    ## Create plots to store waveforms
    if show_plots:
        fi0 = plt.figure(pulse_file+"_a")
        ax0 = fi0.gca()
        ax0.set_xlabel("Time [ms]")
        ax0.set_ylabel(r"$\log_{10}|S_{21}|$")
        if PHASE:
            ax0.set_ylabel(r"$\arg (S_{21})$")
        ax0.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))

        fi1 = plt.figure(pulse_file+"_b")
        ax1 = fi1.gca()
        ax1.set_xlabel(r"$\Re(S_{21})$")
        ax1.set_ylabel(r"$\Im(S_{21})$")
        ax1.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))
    
    ## Start the loop over pulse windows
    k=0
    for pulse_i in range(pulse_start,int(p_params["total_pulses"]),1):

        ## Skip this pulse if it's bad
        if k in bad_pls_idx_arr:
            k+=1
            continue
        
        ## Define the sample index where this pulse window ends
        pulse_i_end = int((pulse_i+1)*samples_per_pulse)
        
        ## Define the edges of the pulse window
        pulse_idx_start = pulse_i_end - N
        pulse_idx_end   = pulse_i_end
        
        ## Grab the timestreams in the various regions
        full_pulse_chunk  = pulse_noise[pulse_idx_start:pulse_idx_end,:]
        pre_trigger_chunk = pulse_noise[pulse_idx_start:pulse_idx_start+pretrig,:]
        post_pulse_chunk  = pulse_noise[pulse_idx_start+pstpuls:pulse_idx_end,:]
        peak_pulse_chunk  = pulse_noise[pulse_idx_start+plstrig-4:pulse_idx_start+plstrig+4]

        ## Determine the two quadratures we care about 
        for q in qdrtrs:

            if q=='phase':
                full_win = np.angle(full_pulse_chunk[:,0])
                pre_trig = np.angle(pre_trigger_chunk[:,0])
                post_pls = np.angle(post_pulse_chunk[:,0])
                peak_reg = np.angle(peak_pulse_chunk[:,0])
                ## Plot the full pulse window against time
                if show_plots and PHASE:
                    ax0.plot(t*1e3,np.angle(full_pulse_chunk[:,0]),alpha=0.25)
            else:
                full_win = np.log10(abs(full_pulse_chunk[:,0]))
                pre_trig = np.log10(abs(pre_trigger_chunk[:,0]))
                post_pls = np.log10(abs(post_pulse_chunk[:,0]))
                peak_reg = np.log10(abs(peak_pulse_chunk[:,0]))
                ## Plot the full pulse window against time
                if show_plots and not PHASE:
                    ax0.plot(t*1e3,np.log10(abs(full_pulse_chunk[:,0])),alpha=0.25)

            ## Calculate the RQs for this pulse window
            m0 = np.mean(pre_trig)   ## Find mean of pre-trigger window
            s0 = np.std( pre_trig)   ## Find sdev of pre-trigger window
            m1 = np.mean(post_pls)   ## Find mean of post-pulse window
            s1 = np.std( post_pls)   ## Find sdev of post-pulse window
            x  = np.max( full_win)   ## Find maximum pulse height in full window
            x0 = np.max( pre_trig)   ## Find maximum pulse height in pre-pulse window
            x1 = np.max( post_pls)   ## Find maximum pulse height in post-pulse window
            p  = np.max( peak_reg)   ## Find maximum pulse height in a window right around the trigger
            a  = np.argmax(full_win) ## Fin the sample with the maximum height in the full window

            ## Keep a running average of the baseline noise in pre-trigger region across all pulse regions
            if PHASE and (q=='phase'):
                noise_averages += m0
            if not PHASE and (q=='logmag'):
                noise_averages += m0

            ## Append our RQs to our lists
            rqs[q][RQ_names[0]].append(m0) ; rqs[q][RQ_names[1]].append(s0)
            rqs[q][RQ_names[2]].append(m1) ; rqs[q][RQ_names[3]].append(s1)
            rqs[q][RQ_names[4]].append( x) ; rqs[q][RQ_names[5]].append( a)
            rqs[q][RQ_names[6]].append(x0) ; rqs[q][RQ_names[7]].append(x1) ; rqs[q][RQ_names[8]].append(p)

        ## Increment pulse counter
        k+=1

        if show_plots:
            ax1.scatter(full_pulse_chunk[:,0].real,full_pulse_chunk[:,0].imag,alpha=0.25)
    
    ## Average the baseline mean over every pulse window
    noise_averages /= (k- (0 if bad_pls_idx_arr[0] is None else len(bad_pls_idx_arr)))
    
    ## Create plots that inform our cuts
    if PHASE:
        bl_means = rqs['phase'][RQ_names[0]]
        bl_sdevs = rqs['phase'][RQ_names[1]]
        pls_maxs = rqs['phase'][RQ_names[4]]
    else:
        bl_means = rqs['logmag'][RQ_names[0]]
        bl_sdevs = rqs['logmag'][RQ_names[1]]
        pls_maxs = rqs['logmag'][RQ_names[4]]

    if show_plots:

        ## Draw some lines to mark the pulse window regions
        if PHASE:
            ax0.axhline(y=noise_averages,color="k",ls='--')
        else:
            ax0.axhline(y=noise_averages,color="k",ls='--')
        ax0.axvline(x=pretrig_seconds*1e3,color="k",ls=':')

        fi2 = plt.figure(pulse_file+"_c")
        ax2 = fi2.gca()
        ax2.hist(bl_means, 
             bins=np.arange(
                 start = np.min( bl_means ) ,
                 stop  = np.max( bl_means ) + 5e-4,
                 step  = 5e-4))
        ax2.axvline(x=np.percentile(bl_means,p1), color="k",ls='--')
        ax2.axvline(x=np.percentile(bl_means,p2), color="k",ls='--')
        if PHASE:
            ax2.set_xlabel(r"Pre-trigger BL mean $\arg(S_{21})$")
        else:
            ax2.set_xlabel(r"Pre-trigger BL mean $\log_{10}(|S_{21}|)$")
        ax2.set_ylabel("Occurences")
        ax2.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))

        fi3 = plt.figure(pulse_file+"_d")
        ax3 = fi3.gca()
        ax3.hist(bl_sdevs, 
             bins=np.arange(
                 start = np.min( bl_sdevs ) ,
                 stop  = np.max( bl_sdevs ) + 1e-5,
                 step  = 1e-5))
        ax3.axvline(x=np.percentile(bl_sdevs,p1), color="k",ls='--')
        ax3.axvline(x=np.percentile(bl_sdevs,p2), color="k",ls='--')
        if PHASE:
            ax3.set_xlabel(r"Pre-trigger BL sdev $\arg(S_{21})$")
        else:
            ax3.set_xlabel(r"Pre-trigger BL sdev $\log_{10}(|S_{21}|)$")
        ax3.set_ylabel("Occurences")
        ax3.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))
        
        fi4 = plt.figure(pulse_file+"_e")
        ax4 = fi4.gca()
        ax4.hist(pls_maxs, 
             bins=np.arange(
                 start = np.min( pls_maxs ) ,
                 stop  = np.max( pls_maxs ) + 5e-4 ,
                 step  = 5e-4))
        ax4.axvline(x=np.percentile(pls_maxs,p1), color="k",ls='--')
        ax4.axvline(x=np.percentile(pls_maxs,p2), color="k",ls='--')
        if PHASE:
            ax4.set_xlabel(r"Full window maximum $\arg(S_{21})$")
        else:
            ax4.set_xlabel(r"Full window maximum $\log_{10}(|S_{21}|)$")
        ax4.set_ylabel("Occurences")
        ax4.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))

    ## Clean up and return the arrays of cut criteria for each pulse window
    del pulse_noise
    return rqs

## Create a plot for each pulse arrival window in the timestream and define a pre-trigger 
## region in which we collect statistics on which to do cuts to remove bad windows
## ARGUMENTS
## 	- LED_files			<array of string>	Each entry is full path to the h5 file containig pulse data
## 	- noise_file		<string>			Full path to the h5 file containig noise data
## 	- vna_file			<string>			Full path to the h5 file containig VNA data
##	- p_params			<dictionary>		Pulse parameter dictionary returned by parse_metadata
##	- p1				<float>				Percentile to draw low-end cut line on distribution plots
##	- p2				<float>				Percentile to draw high-end cut line on distribution plots
##	- decimate_down_to	<float>				Maximum frequency desired in PSDs (in Hz)
##	- pulse_cln_dec		<int>				An explicit decimation factor, used instead of decimate_down_to
## 	- PHASE				<bool>				Whether to plot timestreams in phase or log-mag
##	- show_plots		<bool>				Flag to render plots
## RETURNS
## - pulse_rqs                  dict containing dicts of RQ values, one key per file
##	- mean_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline mean for each pulse window
##	- sdev_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline sdev for each pulse window
##	- maxv_dict			<dictionary>		Each key is an LED file and the item is an array containig the maximum value for each pulse window
def plot_all_pulse_windows(LED_files, noise_file, vna_file, p_params, bad_pls_idxs=None, p1=5, p2=90, decimate_down_to=5e4, pulse_cln_dec=None, PHASE=True, show_plots=False,):
    
    ## Create a dictionary to store the RQ results, one key per file
    pulse_rqs = {}

    ## Loop over every file provided
    for pulse_file in LED_files:

        ## Save the cut criteria to our dictionaries
        pulse_rqs[pulse_file] = plot_pulse_windows(
            pulse_file, noise_file, vna_file, p_params, 
            bad_pls_idx_arr=[None] if bad_pls_idxs is None else bad_pls_idxs[pulse_file],
            p1=p1, p2=p2, decimate_down_to=decimate_down_to, pulse_cln_dec=pulse_cln_dec, 
            PHASE=PHASE, show_plots=show_plots)

    ## Return the cut dictionaries
    return pulse_rqs

## Create the dataframe that 
## ARGUMENTS
## 	- LED_files			<array of string>	Each entry is full path to the h5 file containig pulse data
##	- mean_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline mean for each pulse window
##	- sdev_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline sdev for each pulse window
##	- maxv_dict			<dictionary>		Each key is an LED file and the item is an array containig the maximum value for each pulse window
##	- p1				<float>				Percentile to draw low-end cut line on distribution plots
##	- p2				<float>				Percentile to draw high-end cut line on distribution plots
##	- force_save		<bool>				Force a new cut definition file to be written
## RETURNS
## 	- cut_df			<dataframe>			Dataframe containing min/max cut values for each LED file
def define_default_cuts(LED_files, pulse_RQs, PHASE=True, p1=5, p2=90, force_save=False):
    ## Define a file path and name where cut limits will be stored
    save_path = "/".join(LED_files[0].split("/")[:5])
    series    = save_path.split("/")[-1]
    save_name = series + "_bl_cutvals" 
    save_key  = series+"_cuts"
    if PHASE:
        save_name += "_phase" 
        save_key  += "_phase"

    q = 'phase' if PHASE else 'logmag'

    ## Check if cuts already exist
    if ( os.path.exists(os.path.join(save_path,save_name+".h5")) ) and not force_save:
        cut_df = pd.read_hdf(os.path.join(save_path,save_name+".h5"), key=save_key)
        return cut_df
    elif ( os.path.exists(os.path.join(save_path,save_name+".csv")) ) and not force_save:
        cut_df = pd.read_csv(os.path.join(save_path,save_name+".csv"))
        return cut_df
    else:
        
        ## Create a pandas dataframe for the cut limits
        cut_df = pd.DataFrame(index=LED_files,columns=None)

        ## Define the columns we'll use to store cut limits
        cut_df["sdev_min"] = np.ones(len(LED_files))
        cut_df["sdev_max"] = np.ones(len(LED_files))
        
        cut_df["mean_min"] = np.ones(len(LED_files))
        cut_df["mean_max"] = np.ones(len(LED_files))

        # Now populate each row in the dataframe
        _i = 0
        for _i in np.arange(len(LED_files)):
            cut_df["mean_min"].loc[LED_files[_i]] = np.percentile(pulse_RQs[LED_files[_i]][q]["pre_trig_bl_mean"],p1)
            cut_df["mean_max"].loc[LED_files[_i]] = np.percentile(pulse_RQs[LED_files[_i]][q]["pre_trig_bl_mean"],p2)
            cut_df["sdev_min"].loc[LED_files[_i]] = np.percentile(pulse_RQs[LED_files[_i]][q]["pre_trig_bl_sdev"],p1) 
            cut_df["sdev_max"].loc[LED_files[_i]] = np.percentile(pulse_RQs[LED_files[_i]][q]["pre_trig_bl_sdev"],p2) 

        print("Saving cuts to file", os.path.join(save_path,save_name))
        cut_df.to_hdf( os.path.join(save_path,save_name+".h5") , save_key)
        # cut_df.to_csv( os.path.join(save_path,save_name+".csv"))
            
    return cut_df

def update_cut_limit(cut_df, LED_files, idx, key, value):
    ## Update the key, item pair for this index
    cut_df[key].loc[LED_files[idx]] = value
    return cut_df

def save_cut_df(cut_df, LED_files, PHASE=True):
    ## Define a file path and name where cut limits will be stored
    save_path = "/".join(LED_files[0].split("/")[:5])
    series    = save_path.split("/")[-1]
    save_name = series + "_bl_cutvals" 
    save_key  = series+"_cuts"
    if PHASE:
        save_name += "_phase" 
        save_key  += "_phase"

    print("Saving cuts to file", os.path.join(save_path,save_name))
    cut_df.to_hdf( os.path.join(save_path,save_name+".h5") , save_key)

## Find the indeces for pulse windows, by file, that should be removed from analysis
## ARGUMENTS
##  - pulse_file        <string>            Full path to the h5 file containig pulse data
##  - cut_df            <dataframe>         Dataframe containing min/max cut values for each LED file
##  - mean_dict         <dictionary>        Each key is an LED file and the item is an array containig the pre-trig baseline mean for each pulse window
##  - sdev_dict         <dictionary>        Each key is an LED file and the item is an array containig the pre-trig baseline sdev for each pulse window
##  - maxv_dict         <dictionary>        Each key is an LED file and the item is an array containig the maximum value for each pulse window
## RETURNS
##  - bad_pulses        <array of int>      Array containing the indeces of windows which should be removed
def get_bad_pulse_idxs(pulse_file, cut_df, pulse_rqs, z_pre=3.5, z_post=4.0, z_full=5.0, PHASE=True, verbose=False):
    ## Extract the cut criteria limits
    bl_mean_min = cut_df["mean_min"].loc[pulse_file]
    bl_mean_max = cut_df["mean_max"].loc[pulse_file]
    bl_sdev_min = cut_df["sdev_min"].loc[pulse_file]
    bl_sdev_max = cut_df["sdev_max"].loc[pulse_file]

    ## Pick which quadrature we care about
    q  = 'phase' if PHASE else 'logmag'
    
    ## Extract the cut criteria dictionaries
    bl_means_pre = pulse_rqs[pulse_file][q][RQ_names[0]]
    bl_sdevs_pre = pulse_rqs[pulse_file][q][RQ_names[1]]
    
    bl_means_pst = pulse_rqs[pulse_file][q][RQ_names[2]]
    bl_sdevs_pst = pulse_rqs[pulse_file][q][RQ_names[3]]
    
    pls_maxs  = pulse_rqs[pulse_file][q][RQ_names[4]]
    # pls_amaxs = pulse_rqs[pulse_file][q][RQ_names[5]]
    pre_maxs  = pulse_rqs[pulse_file][q][RQ_names[6]]
    pst_maxs  = pulse_rqs[pulse_file][q][RQ_names[7]]
    pk_maxs   = pulse_rqs[pulse_file][q][RQ_names[8]]
    
    ## Create an empty array for the bad pulse indeces
    bad_pulses = np.array([])
    
    ## Loop over pulse windows
    for k in np.arange(len(bl_means_pre)):
        
        ## Check the cuts for pre-pulse baseline mean
        if (bl_means_pre[k] < bl_mean_min) or (bl_means_pre[k] > bl_mean_max):
            bad_pulses = np.append(bad_pulses, k)
            continue
            
        ## Check the cuts for pre-pulse  baseline sdev
        if (bl_sdevs_pre[k] < bl_sdev_min) or (bl_sdevs_pre[k] > bl_sdev_max):
            bad_pulses = np.append(bad_pulses, k)
            continue
            
        ## Check that no point in post-pulse region is more than +/-Nx RMS from post-pulse baseline mean
        if (np.abs(pst_maxs[k]-bl_means_pst[k]) > z_post*bl_sdevs_pst[k]):
            bad_pulses = np.append(bad_pulses, k)
            continue
        
        ## Check that no point in pre -pulse region is more than +/-Nx RMS from pre -pulse baseline mean
        if (np.abs(pre_maxs[k]-bl_means_pre[k]) > z_pre*bl_sdevs_pre[k]):
            bad_pulses = np.append(bad_pulses, k)
            continue
            
        ## Check that the maximum occurs in the right window, or that no point in full wf is 
        ## more than +/-Nx RMS from pre -pulse baseline mean
        if (pls_maxs[k] != pk_maxs[k]):
            if (np.abs(pls_maxs[k]-bl_means_pre[k]) < z_full*bl_sdevs_pre[k]):
                continue
            bad_pulses = np.append(bad_pulses, k)
            continue

    if verbose:
        print(pulse_file.split('/')[-1], ":", len(bad_pulses), "bad pulses", 
              str(int(1e3*len(bad_pulses)/len(bl_means_pre))/10)+"%")
    return bad_pulses

## Find the indeces for pulse windows, by file, that should be removed from analysis
## ARGUMENTS
## 	- LED_files			<array of string>	Each entry is full path to the h5 file containig pulse data
## 	- cut_df			<dataframe>			Dataframe containing min/max cut values for each LED file
##	- mean_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline mean for each pulse window
##	- sdev_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline sdev for each pulse window
##	- maxv_dict			<dictionary>		Each key is an LED file and the item is an array containig the maximum value for each pulse window
## RETURNS
##	- bad_pls_idxs		<dictionary>		Each key is an LED file and the item is an array containing the indeces of windows which should be removed
def get_all_bad_pulse_idxs(file_list, cut_df, pulse_rqs, z_pre=3.5, z_post=4.0, z_full=5.0, PHASE=True, verbose=False):
    ## Create a dictionary that will contain arrays of bad pulse indeces
    bad_pls_idxs = {}

    ## Loop over every file (LED voltage)
    for pulse_file in file_list:
        
        bad_pls_idxs[pulse_file] = get_bad_pulse_idxs(pulse_file, cut_df, pulse_rqs, 
            z_pre=z_pre, z_post=z_post, z_full=z_full, PHASE=PHASE, verbose=verbose)

    return bad_pls_idxs



def clean_pulse_windows(pulse_file, noise_file, vna_file, p_params, bad_pls_idxs, frac_to_keep=0.5, decimate_down_to=5e4, pulse_cln_dec=None, window_shift_seconds=0, PHASE=True, show_plots=False, verbose=False):
    print('===================')
    print('cleaning pulse file:',pulse_file)
    print('using VNA file:     ',vna_file)
    print('using noise file:   ',noise_file)

    ## Get the decimated timestream and frequency step
    pulse_noise, pulse_info, _, _, _, _, samp_rate = get_decimated_timestream(pulse_file, p_params, decimate_down_to, pulse_cln_dec)

    ## Create a new array of of frequency space with the applied decimation
    samples_per_pulse = int(p_params['time_btw_pulse']*samp_rate)
    N = int(frac_to_keep * samples_per_pulse) ## We look at the second half of a pulse window only
    T = N/samp_rate
    t,f = Prf.build_t_and_f(N,samp_rate)

    time = 1e3*(p_params["time_btw_pulse"]-t[::-1])
    
    ## Define the regions where pulses exist
    ## =====================================
    
    ## This defines where (in # of pulse windows) to start looking for pulse windows
    pulse_start = int(p_params["total_pulses"] * p_params["blank_fraction"])
    if verbose:
        print("Starting pulse partitioning after", pulse_start, "windows (of",p_params["total_pulses"],")")
    
    ## How many samples to shift the pulse window definition
    window_shift = int(window_shift_seconds * samp_rate)
    if verbose:
        print("Shifting pulse window by", window_shift, "samples")
    
    ## Create empty arrays to store our results in
    noise_averages = np.zeros((3),dtype=np.complex128)
    J_r = np.zeros((N,3)); J_arc = np.zeros((N,3))
    
    ## Create empty arrays to store values for histograms
    bl_means = np.array([],dtype=np.complex128)
    bl_sdevs = np.array([])#,dtype=np.complex128)
    
    ## Create a plot to store waveforms
    if show_plots:
        fi0 = plt.figure(pulse_file+"_a")
        ax0 = fi0.gca()
        ax0.set_xlabel("Time [ms]")
        ax0.set_ylabel(r"$\log_{10}|S_{21}|$")
        ax0.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))
        
        fi1 = plt.figure(pulse_file+"_b")
        ax1 = fi1.gca()
        ax1.set_xlabel(r"$\Re(S_{21})$")
        ax1.set_ylabel(r"$\Im(S_{21})$")
        ax1.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))

    ## Count how many good pulses there are
    n_good_pulses = p_params["num_pulses"] - len(bad_pls_idxs[pulse_file])
    
    ## Start the loop over pulse windows
    k=0
    for pulse_i in range(pulse_start,int(p_params["total_pulses"]),1):
        
        ## Skip the bad pulse windows
        if k in bad_pls_idxs[pulse_file]:
            ## Increment the counter
            k += 1
            continue
        
        ## Define the sample index where this pulse window ends
        pulse_i_end = int((pulse_i+1)*samples_per_pulse)
        
        ## Define the start of the pulse free region (period after pulse, before the next one, where it should be baseline noise)
        no_pulse_idx_start = pulse_i_end + window_shift - N 
        
        ## Define the end of the window (where the pulse-free region ends)
        no_pulse_idx_end   = pulse_i_end + window_shift
        
        ## Grab the timestream in that region and average it
        no_pulse_chunk = pulse_noise[no_pulse_idx_start:no_pulse_idx_end,:]
        
        ## Calculate some means and stdevs of this pulse-free timestream
        m = np.mean(no_pulse_chunk,axis=0,dtype=np.complex128) ; bl_means = np.append(bl_means,m[0])
        
        if PHASE:
            s = np.std( np.angle(    no_pulse_chunk[:,0])  ) ; bl_sdevs = np.append(bl_sdevs,s)
        else:
            s = np.std( np.log10(abs(no_pulse_chunk[:,0])) ) ; bl_sdevs = np.append(bl_sdevs,s)
        
        ## Keep a running average of the noise across all pulse regions
        noise_averages += m / n_good_pulses    
        
        ## Plot the pulse free region against time
        if show_plots: # and (k==0):
            ax0.plot(time, np.log10(abs(no_pulse_chunk[:,0])),alpha=0.25)
            ax1.scatter(no_pulse_chunk[:,0].real,no_pulse_chunk[:,0].imag,alpha=0.25)

        ## Convert to the electronics basis and compute the J objects
        r_chunk,arc_chunk,_,_= Prf.electronics_basis(no_pulse_chunk)
        J_r += abs(Prf.discrete_FT(r_chunk))**2 / n_good_pulses * 2 * T
        J_arc += abs(Prf.discrete_FT(arc_chunk))**2 / n_good_pulses * 2 * T
        
        ## Increment the counter
        k += 1
    
    if verbose:
        print("Searched",n_good_pulses,"pulse windows")
        print('used ' + str(n_good_pulses) + ' chunks to find quiescent point')
    
    if show_plots:
        ax0.axhline(y=np.log10(abs(noise_averages[0])),color="k",ls='--')
        
        fi2 = plt.figure(pulse_file+"_c")
        ax2 = fi2.gca()
        if PHASE:
            ax2.hist(np.angle(bl_means))
            ax2.set_xlabel(r"Pre-trigger BL mean $\arg(S_{21})$")
        else:
            ax2.hist(np.log10(abs(bl_means)))
            ax2.set_xlabel(r"Pre-trigger BL mean $\log_{10}(|S_{21}|)$")
        ax2.set_ylabel("Occurences")
        ax2.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))
        
        fi3 = plt.figure(pulse_file+"_d")
        ax3 = fi3.gca()
        if PHASE:
            ax3.hist(bl_sdevs)
            ax3.set_xlabel(r"Pre-trigger BL sdev $\arg(S_{21})$")
        else:
            ax3.hist(bl_sdevs)
            ax3.set_xlabel(r"Pre-trigger BL sdev $\log_{10}(|S_{21}|)$")
        ax3.set_ylabel("Occurences")
        ax3.set_title(".".join(pulse_file.split("/")[-1].split(".")[0:-1]))

    ## Pull the two real quantities from the complex timestream averages
    radius_averages = abs(noise_averages)
    angle_averages  = np.angle(noise_averages)
    if verbose:
        print(radius_averages)
        print(angle_averages)

    ## Rotate the timestream by the averange angle, then get the rotated phase timestream
    pulse_timestream_rotated = pulse_noise*np.exp(-1j*angle_averages)
    angle_timestream = np.angle(pulse_timestream_rotated)

    ## Subtract off the average magnitude value and calculate an arc length
    radius = abs(pulse_noise) - radius_averages
    arc    = angle_timestream*radius_averages

    ## Create output containers for the clean timestreams
    radius_clean = np.zeros(radius.shape)
    arc_clean    = np.zeros(arc.shape)

    if verbose:
        print('built radius and arc length timestreams given by quiescent point')
        print(noise_file)
        
    ## Pull the dictionary containing cleaning coefficients from the noise timestream
    _,data_info = PUf.clean_noi(noise_file[:-3]+'_cleaned.h5')

    ## Loop over each tone in the radius timestream
    for t in range(radius.shape[1]):
        ## Pull the coefficients from the noise cleaning
        radius_coefficient = data_info['radius cleaning coefficient'][t]
        arc_coefficient    = data_info['arc cleaning coefficient'][t]

        ## Clean each tone with the off-resonance tones
        if t == 0:
            off_tone_idcs = [1,2]
        elif t == 1:
            off_tone_idcs = [2]
        elif t == 2:
            off_tone_idcs = [1]

        ## Perform the radius cleaning
        off_tone_radius = np.mean(radius[:,off_tone_idcs],axis=1,dtype=np.float64)
        radius_clean[:,t] = radius[:,t] - radius_coefficient*off_tone_radius

        ## Perform the arc length cleaning
        off_tone_arc = np.mean(arc[:,off_tone_idcs],axis=1,dtype=np.float64)
        arc_clean[:,t] = arc[:,t] - arc_coefficient*off_tone_arc

        if verbose: 
            print('cleaned tone ' + str(t))

    ## Subtract off the mean from cleaned radius and arc length timestreams
    radius_clean -= np.mean(radius_clean,axis=0,dtype='float64')
    arc_clean -= np.mean(arc_clean,axis=0,dtype='float64')
    
    ## Save the clean timestreams to a file
    pulse_noise_clean = Prf.save_clean_timestreams(pulse_file,
                                                   radius_averages,
                                                   angle_averages,
                                                   radius_clean,
                                                   arc_clean,
                                                   samp_rate,
                                                   data_info['radius cleaning coefficient'],
                                                   data_info['arc cleaning coefficient'],
                                                   override=True)

    ## Calculate the PSDs for each of the cleaned pulses
    
    ## Create containers for our output PSDs
    J_r_clean = np.zeros((N,3)); J_arc_clean = np.zeros((N,3))
    
    ## Loop over pulses
    k = 0
    for pulse_i in range(pulse_start,int(p_params["total_pulses"]),1):
        ## Skip the bad pulse windows
        if k in bad_pls_idxs[pulse_file]:
            ## Increment the counter
            k += 1
            continue
        
        ## Define the sample index where this pulse window ends
        pulse_i_end = int((pulse_i+1)*samples_per_pulse) 
        
        ## Define the start of the pulse free region (period after pulse, before the next one, where it should be baseline noise)
        no_pulse_idx_start = pulse_i_end + window_shift - N 
        
        ## Define the end of the window (where the pulse-free region ends)
        no_pulse_idx_end   = pulse_i_end + window_shift
        
        ## Grab the timestream in that region
        no_pulse_chunk = pulse_noise_clean[no_pulse_idx_start:no_pulse_idx_end,:]

        ## Convert the pulse-free region to electronics basis
        r_chunk,arc_chunk,_,_= Prf.electronics_basis(no_pulse_chunk)
        
        ## Compute the PSDs
        J_r_clean += abs(Prf.discrete_FT(r_chunk))**2 / n_good_pulses * 2 * T
        J_arc_clean += abs(Prf.discrete_FT(arc_chunk))**2 / n_good_pulses * 2 * T
        
        ## Increment the counter
        k += 1

    ## Trim the output to the positive frequency region only
    J_r = J_r[f>=0]; J_r_clean = J_r_clean[f>=0]; J_arc = J_arc[f>=0]; J_arc_clean = J_arc_clean[f>=0]
    
    ## Show the PSDs
    if show_plots:
        fig_0, axes_0 = plt.subplots(2,3,sharex=True,sharey='row',figsize=(5*3,10))
        
        Prf.plot_PSDs(f[f>0],J_r,J_arc,pulse_file,
                      ['radius','arc length'],units=['ADCu','ADCu'],savefig='electronics',
                      data_freqs=pulse_info['search freqs'],
                      P_1_clean=J_r_clean,P_2_clean=J_arc_clean,
                      fig_0=fig_0,axes_0=axes_0)



def clean_all_pulse_windows(LED_files, noise_file, vna_file, p_params, bad_pls_idxs, frac_to_keep=0.5, decimate_down_to=5e4, pulse_cln_dec=None, window_shift_seconds=0, PHASE=True, show_plots=False, verbose=False):
    for pulse_file in LED_files:
        clean_pulse_windows(pulse_file, noise_file, vna_file, p_params, bad_pls_idxs, 
            frac_to_keep=frac_to_keep,
            decimate_down_to=decimate_down_to, pulse_cln_dec=pulse_cln_dec, 
            window_shift_seconds=window_shift_seconds, PHASE=PHASE, 
            show_plots=show_plots, verbose=verbose)


def get_average_pulse(pulse_file, vna_file, p_params, bad_pls_idxs, extra_decimation=1, fraction_to_keep=0.5, window_shift_seconds=0, PHASE=True, save_shape=True, show_plots=False, verbose=False, idx=0):
    print('===================')
    print('averaging pulse file: ' + pulse_file)

    ## Get the VNA data for this set of runs
    f,z = PUf.read_vna(vna_file)

    ## Load the clean pulse data
    clean_pulse_file = pulse_file[:-3] + '_cleaned.h5'
    pulse_noise_clean, data_info = PUf.clean_noi(clean_pulse_file)
    if verbose: 
        print('loaded clean pulse data')        
        print('sampling_rate: ' + str(data_info['sampling_rate']))
    
    ## Determine how many samples are in each pulse window
    samples_per_pulse = data_info['sampling_rate'] * p_params["time_btw_pulse"]

    ## Do extra decimation if needed (1 = no decimation) 
    _, pulse_info = PUf.unavg_noi(pulse_file)
    time = Prf.average_decimate(pulse_info['time'],extra_decimation)
    pulse_noise_clean = Prf.average_decimate(pulse_noise_clean,extra_decimation)
    
    ## Update the samples per window and sampling rate with new decimation
    samples_per_pulse_decimated = int(samples_per_pulse / extra_decimation)
    sampling_rate = data_info['sampling_rate'] / extra_decimation
    if verbose:
        print('further decimation by ' + str(extra_decimation) + ' complete')

    ## Create a container to store our average pulse in complex S21 for this file
    pulse_avg    = np.zeros(int(samples_per_pulse_decimated*fraction_to_keep),dtype=np.complex128)
    
    ## Determine how many samples to shift the window by
    window_shift = int(window_shift_seconds*sampling_rate)
    
    ## Identify the first pulse window after the transient period
    pulse_start  = int(p_params["total_pulses"] * p_params["blank_fraction"])
    
    ## Count how many good pulses there are in this file
    n_good_pulses = p_params["num_pulses"] - len(bad_pls_idxs[pulse_file])
    
    ## Start the loop over pulse windows
    k=0
    for pulse_i in range(pulse_start,int(p_params["total_pulses"]),1):
        
        ## Skip the bad pulse windows
        if k in bad_pls_idxs[pulse_file]:
            ## Increment the counter
            k += 1
            continue

        ## Define the sample index where this pulse window ends
        pulse_idx_start = int((pulse_i  )*samples_per_pulse_decimated) + window_shift
        # pulse_idx_end   = int((pulse_i+1)*samples_per_pulse_decimated) + window_shift -1
        pulse_idx_end   = int(round((pulse_i+fraction_to_keep)*samples_per_pulse_decimated,0)) + window_shift
        
        ## Create a list of indeces corresponding to the samples in this pulse window
        pulse_idx_list = np.arange(pulse_idx_start,pulse_idx_end,1,dtype=int)
        
        ## Average the pulses in each good window
        pulse_avg += pulse_noise_clean[pulse_idx_list,0] / n_good_pulses
        
        ## Increment the counter
        k += 1
    
    ## Create a figure in complex S21 to show VNA, full timestream, and average pulse
    if show_plots:
        fig, axs = plt.subplots(1, 2)
        plt.suptitle(pulse_file.split("/")[-1])
        axs[0].set_xlabel(r"$\Re(S_{21})$")
        axs[0].set_ylabel(r"$\Im(S_{21})$")
        axs[1].set_xlabel("Time [ms]")
        if PHASE:
            axs[1].set_ylabel(r"$\arg (S_{21})$")
        else:
            axs[1].set_ylabel(r"$\log10 |S_{21}|$")
        
        axs[0].plot(pulse_noise_clean[:,0].real,pulse_noise_clean[:,0].imag,ls='',marker='.',alpha=0.1,color='grey')
        axs[0].plot(pulse_avg.real,pulse_avg.imag,color='C'+str(idx % 10),ls='-',marker='.')
        axs[0].plot(z.real,z.imag,color='k',ls='-',marker='.',alpha=1.00)
        
        if PHASE:
            axs[1].plot(time[pulse_idx_list,0]*1e3,np.angle(pulse_noise_clean[pulse_idx_list,0]))
            axs[1].plot(time[pulse_idx_list,0]*1e3,np.angle(pulse_avg))
        else:
            axs[1].plot(time[pulse_idx_list,0]*1e3,np.log10(abs(pulse_noise_clean[pulse_idx_list,0])))
            axs[1].plot(time[pulse_idx_list,0]*1e3,np.log10(abs(pulse_avg)))
        
        width = 50 * np.std(pulse_noise_clean[:,0].real)
        x_c = np.mean(pulse_avg.real)
        y_c = np.mean(pulse_avg.imag)
        axs[0].set_xlim([x_c - width/2., x_c + width/2.])
        axs[0].set_ylim([y_c - width/2., y_c + width/2.])
        axs[0].set_aspect('equal','box')
        
        plt.subplots_adjust(wspace=0.5)
        
    if save_shape:
        with h5py.File(clean_pulse_file, "a") as fyle:
            if 'pulse_shape' in fyle.keys():
                del fyle['pulse_shape']
                print('deleted an old pulse shape')
            fyle.create_dataset('pulse_shape',data = np.asarray(pulse_avg))

    print('Used ' + str(n_good_pulses) + ' pulses to average')

    return pulse_avg, sampling_rate

def get_all_average_pulse(LED_files, vna_file, p_params, bad_pls_idxs, extra_decimation=1, fraction_to_keep=0.5, window_shift_seconds=0, PHASE=True, save_shape=True, show_plots=False, verbose=False):
    for j in np.arange(len(LED_files)):        
        get_average_pulse(LED_files[j], vna_file, p_params, bad_pls_idxs,
            extra_decimation=extra_decimation, fraction_to_keep=fraction_to_keep, 
            window_shift_seconds=window_shift_seconds, save_shape=save_shape, 
            show_plots=show_plots, verbose=verbose, idx=j)
        

def align_all_pulses(LED_files, nse_files, vna_file, sum_file, p_params, charFs, charZs, MB_fit_vals, Voltages, readout_unit='df', fraction_to_keep=0.5, tw_min_us=8000, tw_max_us=10000, data_T_K=10.0e-3, cmap=None):
    
    if cmap is None:
        cmap = plt.get_cmap('OrRd')
    plot_width = -999

    ## Open the cleaned data and pull the data sampling rate, pulse template, and pulse noise
    with h5py.File(LED_files[0][:-3] + '_cleaned.h5', "r") as fyle:
        sampling_rate = np.array(fyle['sampling_rate'])

    ## Define the time window for the pulse-full region, in microseconds
    time_window_range = fraction_to_keep * p_params['time_btw_pulse'] *1e6
    time_window = np.arange(0,time_window_range,1/sampling_rate*1e6)#[:-1]

    ## Create some strings to use as plot titles and handles later
    AWF_string = str(int(10*p_params['pulse_w'])/10) + " us"
    title_1    = 'Power ' + str(p_params['rf_power']) + '; AWF ' + AWF_string + ': pulses in S21'
    title_1p5  = 'Power ' + str(p_params['rf_power']) + '; AWF ' + AWF_string + ': pulses in S21, zoomed in'
    title_1p75 = 'Alignment of pulses using largest pulse' 
    title_2    = 'Average Pulse Shapes (along pulse alignment axis)'
    title_3    = 'Pulse trajectory in Resonator basis' 
    title_4    = 'Pulse trajectory in Quasiparticle basis' 

    ## Create an output dictionary for the maximum pulse amplitude in each file
    max_avgpulse_height = {}

    ## Pull the readout frequency from the characterization data
    ## then get the VNA data for this run
    readout_f = charFs[0,1].real
    f,z = PUf.read_vna(vna_file)

    ## Reorder the files such that the largest LED voltage comes first
    LED_files = np.sort(LED_files)[::-1]
    Voltages  = np.sort(Voltages)[::-1]

    ## Loop over the full list of LED files
    ## Initialize an index to count files as we loop
    i = 0
    for pulse_file in LED_files:
        print('===================')
        print('cleaning pulse file:',pulse_file)
        print('using VNA file:     ',vna_file)
        print('using summary file: ',sum_file)
        char_file = sum_file

        ## Get the cleaned data and average pulse
        clean_pulse_file = pulse_file[:-3] + '_cleaned.h5'
        with h5py.File(clean_pulse_file, "r") as fyle:
            pulse_avg = np.array(fyle["pulse_shape"],dtype=np.complex128)
            pulse_timestream = np.array(fyle["cleaned_data"],dtype=np.complex128)

        ## >TO DO< The following conversions should happen after the rotation and alignment of the cleaned timestreams
        ## >TO DO< Follow up on this - actually, we only want to do the rotation if we're picking some :optimal: readout direction
        ## based in the raw units - I think we've done it right with no rotation so far.

        ## Get the timestreams and average pulse in resonator basis
        df_f, d1_Q, _, _ = Prf.resonator_basis(pulse_avg,readout_f*1e-3,f*1e-3,z,charFs[0].real*1e-3,charZs[0])
        df_f_timestream, d1_Q_timestream, _, _ = Prf.resonator_basis(pulse_timestream[:,0],readout_f*1e-3,f*1e-3,z,charFs[0].real*1e-3,charZs[0])
        
        ## Get the timestreams and average pulse in quasiparticle basis
        dk1, dk2, =                       Prf.quasiparticle_basis(df_f, d1_Q,
                                                                  data_T     = data_T_K, 
                                                                  MB_results = MB_fit_vals,
                                                                  readout_f  = readout_f*1e-3)
        dk1_timestream, dk2_timestream, = Prf.quasiparticle_basis(df_f_timestream, d1_Q_timestream,
                                                                  data_T     = data_T_K, 
                                                                  MB_results = MB_fit_vals,
                                                                  readout_f  = readout_f*1e-3)

        ## Baseline(mean)-subtract the average pulse, then find its stdev
        ## Using the last five samples of the average pulse to get mean, sdev
        pulse_avg_mb = pulse_avg - np.mean(pulse_avg[-5:],dtype=np.complex128)
        std = np.std(abs(pulse_avg_mb[-5:]),dtype=np.complex128)
        
        ## Calculate the average angle of the average pulse in the specified alignment window
        ## Calculate this only for the first (largest) LED file and then rotate all the others to it
        if i==0:
            average_angle = np.mean(np.angle(pulse_avg_mb[np.logical_and(time_window>tw_min_us,time_window<tw_max_us)]))
        
        ## Baseline(mean)-subtract the (raw) pulse timestream for the on-resonance tone
        ## Again using the last five samples of the average pulse to get mean
        pulse_timestream_mb = pulse_timestream[:,0] - np.mean(pulse_avg[-5:],dtype=np.complex128)

        ## Apply the rotation by the average angle to the baseline-subtracted timestream and average pulse
        pulse_avg_rotated = pulse_avg_mb * np.exp(-1j*average_angle)
        pulse_timestream_rotated = pulse_timestream_mb * np.exp(-1j*average_angle)

        ## Define the pulse template we want to use and which timestream to use it on
        ## Also subtract the baseline of the template to make sure baseline=0 for optimal filtering
        
        ## == CHOOSE ONE TO DO ANALYSIS == ##
        readout_units = ["df", "dQ", "dk1", "dk2", "phase", "mag"]
        if readout_unit not in readout_units:
            print(readout_unit, "is not a valid readout unit. Choose from:", readout_units)
            readout_unit = readout_units[0]

        template = None
        full_ts  = None
        ylbl     = None

        ## Fractional change in frequency 
        if readout_unit == "df":
            print("Using df/f readout")

            mean_avg = np.mean(df_f[:20])
            template = df_f - mean_avg
            full_ts  = df_f_timestream - mean_avg
            ylbl     = r"Frequency Shift $\delta f / f$"

        ## Dissipation direction quasiparticle basis
        elif readout_unit == "dQ":
            print("Using d(1/Q) readout")
            
            mean_avg = np.mean(d1_Q[:20])
            template = d1_Q - mean_avg
            full_ts  = d1_Q_timestream - mean_avg
            ylbl     = r"Dissipation shift $\delta (1/Q)$"
        
        ## Frequency direction quasiparticle basis
        elif readout_unit == "dk1":
            print("Using dk1 (qp-dissipation) readout")
            
            mean_avg = np.mean(dk1[:20])
            template = dk1 - mean_avg
            full_ts  = dk1_timestream - mean_avg
            ylbl     = r"Frequency qp shift $\delta \kappa_1$ [$\mu$m$^{-3}$]"

        ## Dissipation direction quasiparticle basis
        elif readout_unit == "dk2":
            print("Using dk2 (qp-frequency) readout")
            
            mean_avg = np.mean(dk2[:20])
            template = dk2 - mean_avg
            full_ts  = dk2_timestream - mean_avg
            ylbl     = r"Dissipation qp shift $\delta \kappa_2$ [$\mu$m$^{-3}$]"

        ## Rotated phase direction
        elif readout_unit == "phase":
            print("Using phase readout")

            mean_avg = np.mean( np.angle(pulse_avg_rotated[:20]) )
            template = np.angle(pulse_avg_rotated) - mean_avg
            full_ts  = np.angle(pulse_timestream_rotated) - mean_avg
            ylbl     = r"Phase shift $\delta \phi$ [rad]"

        ## Rotated log mag direction
        elif readout_unit == "mag":
            print("Using log mag readout")

            mean_avg = np.mean( np.log10(abs(pulse_avg_rotated[:20])) )
            template = np.log10(abs(pulse_avg_rotated)) - mean_avg
            full_ts  = np.log10(abs(pulse_timestream_rotated)) - mean_avg
            ylbl     = r"Log Mag Shift $\delta |R|$ [dB]"

        ## Open the clean data file and save the template and noise
        with h5py.File(clean_pulse_file, "a") as fyle:
            print("Saving clean pulse file:",clean_pulse_file)
            if 'signal_template' in fyle.keys():
                del fyle['signal_template']
            if 'full_timestream' in fyle.keys():
                del fyle['full_timestream']
            fyle.create_dataset('signal_template',data = np.asarray(template)) ## Save the average pulse template in selected units
            fyle.create_dataset('full_timestream',data = np.asarray(full_ts)) ## Save the full pulse timestream in selected units
        
        ## Save the maximum pulse height for this file
        max_avgpulse_height[pulse_file] = np.max(template)

        ## Define some labels to add to plots
        label_c = 'characterization data' if i == 0 else None
        label_V = 'VNA' if i == 0 else None
        label_p = 'pulse data ' if i == 0 else None
        
        ## Grab the vna + average pulse plot, draw the vna, average pulse, and characterization data
        ## we only have to draw the VNA on the first file
        plt.figure(title_1)
        plt.title(title_1)
        plt.plot(pulse_avg.real,pulse_avg.imag,ls='-',marker='.',markersize=5,color='C'+str(i%10))
        if i==0:
            plt.plot(z.real,z.imag,color='k',label=label_V)
        plt.plot(charZs.real,charZs.imag,marker='*',markersize=10,ls='-',label=label_c,zorder=-5*i+200)
        plt.axhline(0,color='grey')
        plt.axvline(0,color='grey')
        
        ## Adjust the plot limits
        width = np.std(pulse_avg.real) * 8e3
        x_c = np.mean(pulse_avg.real)
        y_c = np.mean(pulse_avg.imag)
        plt.axis([x_c - width/2., x_c + width/2., y_c-width/2., y_c+width/2.])
        plt.gca().set_aspect('equal','box')
        plt.legend()

        ## Grab the zoomed-in vna + average pulse plot, draw the vna, average pulse, and characterization data
        ## we only have to draw the VNA on the first file
        plt.figure(title_1p5)
        plt.title(title_1p5)
        plt.plot(pulse_avg.real,pulse_avg.imag,ls='-',marker='.',markersize=5,color='C'+str(i%10),label=label_p,zorder=-5*i+200)
        if i==0:
            plt.plot(z.real,z.imag,color='k',label=label_V)
        
        ## Use the first file to set the extent of the zoomed-in plot
        width = 150 * np.std(pulse_avg.real)
        if width > plot_width:
            plot_width = width
        x_c = np.mean(pulse_avg.real)
        y_c = np.mean(pulse_avg.imag)
        plt.axis([x_c - plot_width/2., x_c + plot_width/2., y_c-plot_width/2., y_c+plot_width/2.])

        ## Grab the average pulse rotation plot
        plt.figure(title_1p75)
        plt.title(title_1p75)
        plt.plot(pulse_avg_rotated.real,pulse_avg_rotated.imag,ls='-',marker='.',markersize=5,color='C'+str(i%10))
        plt.gca().set_aspect('equal', 'box')
        
        ## Grab the average pulse timestream plot and draw the timestream of the average pulse
        plt.figure(title_2)
        plt.xlabel('milliseconds')
        plt.ylabel(ylbl)
        plt.title(title_2)
        plt.plot(time_window/1e3,template,ls='-',marker=None,markersize=5,label=str(Voltages[i])+" V",color=cmap( (Voltages[i]-1.5) / (np.max(Voltages)-1.5) ))#'C'+str(i))
        ax = plt.gca()
        plt.fill_between([tw_min_us/1e3,tw_max_us/1e3], y1=ax.get_xlim()[1], y2=ax.get_xlim()[0], color='k', alpha=0.2)

        ## Grab the average pulse rotation plot
        plt.figure(title_3)
        plt.title(title_3)
        plt.xlabel(r"Frequency Shift $\delta f/f$")
        plt.ylabel(r"Dissipation Shift $\delta (1/Q)$")
        plt.plot(df_f,d1_Q,ls='-',marker='.',markersize=5,color='C'+str(i%10))
        plt.gca().set_aspect('equal', 'box')

        ## Grab the average pulse rotation plot
        plt.figure(title_4)
        plt.title(title_4)
        plt.xlabel(r"Kappa 2 (Frequency) Shift $\delta \kappa_2$ [$\mu$m$^{-3}$]")
        plt.ylabel(r"Kappa 1 (Dissipation) Shift $\delta \kappa_1$ [$\mu$m$^{-3}$]")
        plt.plot(dk2,dk1,ls='-',marker='.',markersize=5,color='C'+str(i%10))
        plt.gca().set_aspect('equal', 'box')

        ## Increment our file counter
        i += 1
        
    ## Grab the zoomed-in vna + average pulse plot
    plt.figure(title_1p5)

    ## Loop over the noise files again
    for ai in np.arange(len(nse_files)):
        ## Get the noise timestream and average
        frequencies_scanned, noise_mean_scanned = PUf.avg_noi(nse_files[ai],time_threshold=15.0*p_params["blank_fraction"])
        ## Draw the points
        plt.plot(noise_mean_scanned[0].real,noise_mean_scanned[0].imag,marker='*',markersize=10,markeredgecolor='k',ls='None',color='y',zorder=-5*i+200,alpha=ai*.2,label="Noise Averages")

    plt.legend()

    plt.figure(title_2)
    plt.legend(loc='best')

    return max_avgpulse_height


def get_pulse_template(template_file, p_params, window_shift_J=0, f_max=1e4, use_fit_as_template=False):

    ## Create titles for the plots
    title      = 'readout power ' + str(p_params['rf_power']) + ': '

    ## Pick the highest LED voltage data to use as signal template
    print("Using file:",template_file,"as pulse template")

    ## Find the clean pulse file, use the last one in the set that's been analyzed so far
    clean_pulse_file = template_file[:-3] + '_cleaned.h5'

    with h5py.File(clean_pulse_file, "r") as fyle:
        pulse_avg = np.array(fyle['signal_template'])
        sampling_rate = np.array(fyle['sampling_rate'])
    ylbl  = r"Frequency Shift $\delta f / f$"

    ## Determine a window size equivalent to the full pulse template window
    N = len(pulse_avg)
    print(N, "samples per window")

    ## Determine total period of the template and create a time-domain array and a freq-domain array
    T = N/sampling_rate
    time, f = Prf.build_t_and_f(N,sampling_rate)

    ## Define an offset function to recenter the pulse
    t_offset = lambda N: int( (p_params['delay_ms']*1e-3) * sampling_rate)
    # t_offset = lambda N: int(N * (2*delay_ms*1e-3/time_btw_pulse) / frac_to_keep)

    ## Define an exponential function
    exponential = lambda x, A, tau: np.heaviside(x-time[t_offset(N)],0.5) * A * expit(-1*(x-time[t_offset(N)])/tau)
    dbl_expA    = lambda x, A, t1, t2: np.heaviside(x-time[t_offset(N)],0.5) * A * ( expit(-1*(x-time[t_offset(N)])/t1) + expit(-1*(x-time[t_offset(N)])/t2) )
    
    ## Fit the average pulse to an exponential (in k2)
    popt, pcov  = curve_fit(exponential,time,pulse_avg,p0=[5.0,       1.24e-3])
    popt2,pcov2 = curve_fit(dbl_expA,time,pulse_avg   ,p0=[5.0,       1.24e-3/5.,1.24e-3*5.])
        
    ## Calculate the best fit curve for the average pulse
    pulse_fit  = exponential(time,*popt)
    pulse_fit2 = dbl_expA(time,*popt2)
        
    ## Numerically integrate the fit, then FT it
    if use_fit_as_template:   
        A = np.trapz(pulse_fit,dx = time[1]-time[0])
        s = Prf.discrete_FT(pulse_fit)
    else:
        A = np.trapz(pulse_avg,dx = time[1]-time[0])
        s = Prf.discrete_FT(pulse_avg)

    ## Define a mask for evaluating fourier space components
    f_mask = np.logical_and(f <= f_max, f >= -1*f_max)
    N_mask = len(f[f_mask])
    f_plot = np.logical_and(f > 0, f <= f_max)
    new_fs = max(f[f_mask])

    ## Get the magnitude of s^2 for the masked region
    S_mag = abs(s[f_mask]**2)

    ## Create a plot to store the template waveform and best fit
    f1p5  = plt.figure()
    ax1p5 = f1p5.gca()
    ax1p5.plot(time*1e3,pulse_fit, 'C7',linewidth=3,label='time constant: {:.2e}ms'.format(popt[-1]*1e3), ls='--' )
    # ax1p5.plot(time*1e3,pulse_fit3,'C6',linewidth=3,label='Float weights\n'+r'$\tau_1=$ {:.2e}ms'.format(popt3[-2]*1e3)+"\n"+r'$\tau_2=$ {:.2e}ms'.format(popt3[-1]*1e3))
    ax1p5.plot(time*1e3,pulse_fit2,'k' ,linewidth=1,label='Equal weights\n'+r'$\tau_1=$ {:.2e}ms'.format(popt2[-2]*1e3)+"\n"+r'$\tau_2=$ {:.2e}ms'.format(popt2[-1]*1e3))
    ax1p5.plot(time*1e3,pulse_avg, label="Average pulse")
    ax1p5.axhline(y=0,color='r',ls='--')
    ax1p5.set_title(title + 'signal pulse timestream and fit')
    ax1p5.set_xlabel("Time [ms]")
    ax1p5.set_ylabel(ylbl)
    # ax1p5.set_ylabel(r"Dissipation $\delta \kappa_2$ [$\mu$m$^{-3}$]")
    plt.legend()

    f1p7  = plt.figure()
    ax1p7 = f1p7.gca()
    ax1p7.plot(time*1e3,pulse_fit -pulse_avg,'C7',linewidth=3,label=r'$\tau=$ {:.2e}ms'.format(popt[-1]*1e3) , ls='--' )
    # ax1p7.plot(time*1e3,pulse_fit3-pulse_avg,'C6',linewidth=3,label='Float weights\n'+r'$\tau_1=$ {:.2e}ms'.format(popt3[-2]*1e3)+"\n"+r'$\tau_2=$ {:.2e}ms'.format(popt3[-1]*1e3))
    ax1p7.plot(time*1e3,pulse_fit2-pulse_avg,'k' ,linewidth=1,label='Equal weights\n'+r'$\tau_1=$ {:.2e}ms'.format(popt2[-2]*1e3)+"\n"+r'$\tau_2=$ {:.2e}ms'.format(popt2[-1]*1e3))
    ax1p7.axhline(y=0,color='r',ls='--')
    ax1p7.set_title(title + 'fit residuals')
    ax1p7.set_xlabel("Time [ms]")
    ax1p7.set_ylabel("Fit residual")
    plt.legend()

    return S_mag, s, A

def get_noise_template(template_file, s, p_params, bad_pls_idxs, window_shift_J=0, f_max=1e4, PHASE=True):

    ## Create titles for the plots
    title = 'readout power ' + str(p_params['rf_power']) + ': '

    print("Using file:",template_file,"to characterize noise")

    ## Find the clean noise file, use the last one in the set that's been analyzed so far
    clean_noise_file = template_file[:-3] + '_cleaned.h5'

    ## Open the cleaned data and pull the data sampling rate, pulse template, and pulse noise
    with h5py.File(clean_noise_file, "r") as fyle:
        sampling_rate = np.array(fyle['sampling_rate'])          
        pulse_noise   = np.array(fyle["full_timestream"])
        pulse_avg     = np.array(fyle["signal_template"])

    ## Determine a window size equivalent to the full noise template window
    ## Note that this must be exactly the same size as the pulse template with 
    ## the same sampling rate
    N = len(pulse_avg)
    del pulse_avg
    print(N, "samples per window")

    ## Determine total period of the template and create a time-domain array and a freq-domain array
    T = N/sampling_rate
    time, f = Prf.build_t_and_f(N,sampling_rate)

    ## Determine how many samples to shift the window when calculating J
    samples_per_pulse  = sampling_rate*p_params['time_btw_pulse']
    window_shift_J_idx = int(window_shift_J*sampling_rate)

    ## Create containers in which to store J and average noise timestream
    J = np.zeros(N)
    avg_noise = np.zeros(N)
        
    ## Count how many good pulses there are in this file
    n_good_pulses = p_params['num_pulses'] - len(bad_pls_idxs[template_file])

    ## This defines where (in # of pulse windows) to start looking for pulse windows
    pulse_start = int(p_params['total_pulses'] * p_params['blank_fraction'])
    samples_per_pulse = int(p_params['time_btw_pulse']*sampling_rate)

    ## Loop over all the pulse windows in the file
    ## We want to skip the windows cut out when finding noise free region
    k = 0
    for pulse_i in range(pulse_start,int(p_params['total_pulses']),1):

        ## Skip the bad pulse windows
        if k in bad_pls_idxs[template_file]:
            ## Increment the counter
            k += 1
            continue

        ## Find the index for the end of this pulse window
        pulse_i_end = int((pulse_i+1)*samples_per_pulse) 

        ## Apply the window shift and an array of samples for the window timestream points
        ## Are we really trying to find a pulse free region? seems like it
        no_pulse_idx_start = pulse_i_end + window_shift_J_idx - N
        no_pulse_idx_end   = pulse_i_end + window_shift_J_idx
        no_pulse_idx_list  = np.arange(no_pulse_idx_start,no_pulse_idx_end,1,dtype=int)
        no_pulse_noise_i   = pulse_noise[no_pulse_idx_list]

        ## Calculate the average noise timestream
        avg_noise += (np.angle(no_pulse_noise_i) if PHASE else np.log10(abs(no_pulse_noise_i))) / n_good_pulses

        ## Caclulate the average J for this region
        J += abs(Prf.discrete_FT(no_pulse_noise_i))**2 / n_good_pulses * 2 * T

        ## Increment the counter
        k += 1
        
    ## Determine the average J for plotting
    J_avg = np.mean(J)

    ## Define a mask for evaluating fourier space components
    f_mask = np.logical_and(f <= f_max, f >= -1*f_max)
    N_mask = len(f[f_mask])
    f_plot = np.logical_and(f > 0, f <= f_max)
    new_fs = max(f[f_mask])

    ## Calculate the denominator for the optimal filter
    denominator = np.sum(abs(s[f_mask])**2/J[f_mask])

    ## Calculate the baseline resolution
    b7_res = np.sqrt(((2*T*denominator)**-1))
    print("Baseline resolution:", b7_res)

    ## Plot the J we'll use for optimal filtering
    f2p0  = plt.figure()
    ax2p0 = f2p0.gca()
    plt.plot(f[f_plot],J[f_plot],color='k')#,zorder=-5*j+5,color='k')
    ax2p0.set_title(title + 'noise frequency domain')
    ax2p0.set_xlabel("Frequency [Hz]")
    ax2p0.set_ylabel(r"$J$")
    ax2p0.set_xscale('log')
    ax2p0.set_yscale('log')
    ax2p0.set_ylim([1e-1*J_avg,5e2*J_avg])

    ## Plot the noise timestream to make sure we got it right
    f2p1  = plt.figure()
    ax2p1 = f2p1.gca()
    ax2p1.plot(time*1e3,avg_noise,color='k')
    ax2p1.set_title(title + 'noise time domain')
    ax2p1.set_xlabel("Time [ms]")
    ax2p1.set_ylabel(r"$\mathrm{arg}(S_{21})$ [rad]" if PHASE else r"$\log10 |S_{21}|$ [dBc]" )

    return J_avg, denominator, b7_res
