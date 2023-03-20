import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import PyMKID_USRP_functions as PUf
import PyMKID_resolution_functions as Prf
import TimestreamHelperFunctions as Thf

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
    rf_power = md['power']
    
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

    return pulse_noise, N, T, t, f, pulse_fs


# ## Given some waveform and pulse arrival parameters, pull a specific pulse window ROI
# ## By default the ROI starts at the first sample in the window and extends the full window
# ## The user can shrink or expand the window in time, as well as shift it relative to the start of the window
# def get_pulse_window(waveform, pls_window_idx, samples_per_pulse, sampling_rate, frac_to_keep=1.0, offset_sec=0.0):

#     ## Define the sample index where this pulse window ends
#     window_i_end  = int((pls_window_idx+1)*samples_per_pulse)

#     ## Define how many samples the ROI is
#     N_samps_roi   = int(frac_to_keep*samples_per_pulse) - 1

#     ## Caclulate how many samples by which to shift the window
#     N_samps_shift = int(offset_sec*sampling_rate)
    
#     ## Define the edges of the pulse window
#     pulse_idx_start = pulse_i_end + N_samps_shift - N_samps_roi 
#     pulse_idx_end   = pulse_i_end + N_samps_shift

#     return waveform[pulse_idx_start:pulse_idx_end,:]



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
##	- mean_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline mean for each pulse window
##	- sdev_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline sdev for each pulse window
##	- maxv_dict			<dictionary>		Each key is an LED file and the item is an array containig the maximum value for each pulse window
def plot_pulse_windows(LED_files, noise_file, vna_file, p_params, 
    p1=5, p2=90, decimate_down_to=5e4, pulse_cln_dec=None,
    PHASE=True, show_plots=False,):
    
    ## Define the time that separates "pre-trigger" region from rest of pulse window
    pretrig_seconds = (p_params["delay_ms"]-0.25)*1e-3
    
    ## Create dictionaries to store the cut criteria parameters
    mean_dict = {}
    sdev_dict = {}
    maxv_dict = {}

    ## Loop over every file provided
    for pulse_file in LED_files:
        print('===================')
        print('plotting pulse file:',pulse_file)
        print('using VNA file:     ',vna_file)
        print('using noise file:   ',noise_file)

        ## Get the decimated timestream and frequency step
        pulse_noise, N, T, t, f, samp_rate = get_decimated_timestream(pulse_file, p_params, decimate_down_to, pulse_cln_dec)

        ## Define the regions where pulses exist
        ## =====================================
        
        ## This defines where (in # of pulse windows) to start looking for pulse windows
        pulse_start = int(p_params["total_pulses"] * p_params["blank_fraction"])
        samples_per_pulse = int(p_params["time_btw_pulse"]*samp_rate)
        
        ## How many samples to shift the pulse window definition
        pretrig = int(pretrig_seconds * samp_rate)
        
        ## Create an empty array to store our results in
        ## This is the average baseline of the three timestreams across all pulse windows
        noise_averages = 0 # np.zeros((3),dtype=np.complex128)
        
        ## Create empty arrays to store values which we will use to perform quality cuts
        ## These will have an entry for each pulse window
        bl_means = np.array([])#,dtype=np.complex128)
        bl_sdevs = np.array([])#,dtype=np.complex128)
        pls_maxs = np.array([])#,dtype=np.complex128)
        
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
            
            ## Define the sample index where this pulse window ends
            pulse_i_end = int((pulse_i+1)*samples_per_pulse)
            
            ## Define the edges of the pulse window
            pulse_idx_start = pulse_i_end - N
            pulse_idx_end   = pulse_i_end
            
            ## Grab the timestream in that region and average it
            full_pulse_chunk  = pulse_noise[pulse_idx_start:pulse_idx_end,:]
            pretrig_pls_chunk = pulse_noise[pulse_idx_start:pulse_idx_start+pretrig,:]
            
            ## Determine the two quadratures we care about
            phase  = np.angle(pretrig_pls_chunk[:,0])
            logmag = np.log10(abs(pretrig_pls_chunk[:,0]))
        
            ## Find the mean, sdev of pre-trigger window and maximum pulse height in full window
            if PHASE:
                m = np.mean(phase ) ; bl_means = np.append(bl_means,m)
                s = np.std( phase ) ; bl_sdevs = np.append(bl_sdevs,s)
                x = np.max(np.angle(full_pulse_chunk[:,0])) ; pls_maxs = np.append(pls_maxs,x)
            else:
                m = np.mean(logmag) ; bl_means = np.append(bl_means,m)
                s = np.std( logmag) ; bl_sdevs = np.append(bl_sdevs,s)
                x = np.max(np.log10(abs(full_pulse_chunk[:,0]))) ; pls_maxs = np.append(pls_maxs,x)
            
            ## Keep a running average of the baseline noise in pre-trigger region across all pulse regions
            noise_averages += m    
            
            ## Plot the full pulse window against time
            if show_plots:
                if PHASE:
                    ax0.plot(t*1e3,np.angle(full_pulse_chunk[:,0]),alpha=0.25)
                else:
                    ax0.plot(t*1e3,np.log10(abs(full_pulse_chunk[:,0])),alpha=0.25)
                ax1.scatter(full_pulse_chunk[:,0].real,full_pulse_chunk[:,0].imag,alpha=0.25)
            
            ## Increment the good pulse counter
            k += 1
        
        ## Average the baseline mean over every pulse window
        noise_averages /= k
        
        ## Save the cut criteria to our dictionaries
        mean_dict[pulse_file] = bl_means
        sdev_dict[pulse_file] = bl_sdevs
        maxv_dict[pulse_file] = pls_maxs
             
        ## Draw some lines to mark the pulse window regions
        if show_plots:
            if PHASE:
                ax0.axhline(y=noise_averages,color="k",ls='--')
            else:
                ax0.axhline(y=noise_averages,color="k",ls='--')
            ax0.axvline(x=pretrig_seconds*1e3,color="k",ls=':')
        
        ## Create plots that inform our cuts
        if show_plots:
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

    ## Return the cut dictionaries
    return mean_dict, sdev_dict, maxv_dict

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
def define_default_cuts(LED_files, mean_dict, sdev_dict, maxv_dict, PHASE=True, p1=5, p2=90, force_save=False):
    ## Define a file path and name where cut limits will be stored
    save_path = "/".join(LED_files[0].split("/")[:5])
    series    = save_path.split("/")[-1]
    save_name = series + "_bl_cutvals" 
    save_key  = series+"_cuts"
    if PHASE:
        save_name += "_phase" 
        save_key  += "_phase"

    ## Check if cuts already exist
    if ( os.path.exists(os.path.join(save_path,save_name+".h5")) ) and not force_save:
        cut_df = pd.read_hdf(os.path.join(save_path,save_name+".h5"), key=save_key)
        save_cuts = False
        print("H5 cuts file exists, not overwriting...")
    elif ( os.path.exists(os.path.join(save_path,save_name+".csv")) ) and not force_save:
        cut_df = pd.read_csv(os.path.join(save_path,save_name+".csv"))
        save_cuts = False
        print("CSV cuts file exists, not overwriting...")
    else:
        save_cuts = True
        print("Saving new cut definitions to:",save_name)
        
        ## Create a pandas dataframe for the cut limits
        cut_df = pd.DataFrame(index=LED_files,columns=None)

        ## Define the columns we'll use to store cut limits
        cut_df["mean_min"] = np.ones(len(LED_files))
        cut_df["mean_max"] = np.ones(len(LED_files))
        cut_df["sdev_min"] = np.ones(len(LED_files))
        cut_df["sdev_max"] = np.ones(len(LED_files))
        cut_df["wfmx_min"] = np.array([None] * len(LED_files))
        cut_df["wfmx_max"] = np.array([None] * len(LED_files))

        ## Now populate each row in the dataframe (one entry per LED file)
        _i = 0
        for _i in np.arange(len(LED_files)):
            cut_df["mean_min"].loc[LED_files[_i]] = np.percentile(mean_dict[LED_files[_i]],p1)
            cut_df["mean_max"].loc[LED_files[_i]] = np.percentile(mean_dict[LED_files[_i]],p2)
            cut_df["sdev_min"].loc[LED_files[_i]] = np.percentile(sdev_dict[LED_files[_i]],p1) 
            cut_df["sdev_max"].loc[LED_files[_i]] = np.percentile(sdev_dict[LED_files[_i]],p2) 
            cut_df["wfmx_max"].loc[LED_files[_i]] = None
            cut_df["wfmx_max"].loc[LED_files[_i]] = None
        
        if (save_cuts or force_save):
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
## 	- LED_files			<array of string>	Each entry is full path to the h5 file containig pulse data
## 	- cut_df			<dataframe>			Dataframe containing min/max cut values for each LED file
##	- mean_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline mean for each pulse window
##	- sdev_dict			<dictionary>		Each key is an LED file and the item is an array containig the pre-trig baseline sdev for each pulse window
##	- maxv_dict			<dictionary>		Each key is an LED file and the item is an array containig the maximum value for each pulse window
## RETURNS
##	- bad_pls_idxs		<dictionary>		Each key is an LED file and the item is an array containing the indeces of windows which should be removed
def get_bad_pulse_idxs(LED_files, cut_df, mean_dict, sdev_dict, maxv_dict):
    ## Create a dictionary that will contain arrays of bad pulse indeces
    bad_pls_idxs = {}

    ## Loop over every file (LED voltage)
    for pulse_file in LED_files:
        
        ## Extract the cut criteria limits
        bl_mean_min = cut_df["mean_min"].loc[pulse_file]
        bl_mean_max = cut_df["mean_max"].loc[pulse_file]
        bl_sdev_min = cut_df["sdev_min"].loc[pulse_file]
        bl_sdev_max = cut_df["sdev_max"].loc[pulse_file]
        wf_max__min = cut_df["wfmx_min"].loc[pulse_file]
        wf_max__max = cut_df["wfmx_max"].loc[pulse_file]
        
        ## Extract the cut criteria dictionaries
        bl_means = mean_dict[pulse_file]
        bl_sdevs = sdev_dict[pulse_file]
        pls_maxs = maxv_dict[pulse_file]
        
        ## Create an empty array for the bad pulse indeces
        bad_pulses = np.array([])
        
        ## Loop over pulse windows
        for k in np.arange(len(bl_means)):
            
            ## Check the cuts for baseline mean
            if (bl_means[k] < bl_mean_min) or (bl_means[k] > bl_mean_max):
                bad_pulses = np.append(bad_pulses, k)
                continue
                
            ## Check the cuts for baseline sdev
            if (bl_sdevs[k] < bl_sdev_min) or (bl_sdevs[k] > bl_sdev_max):
                bad_pulses = np.append(bad_pulses, k)
                continue
                
            if wf_max__max is not None:
                if (pls_maxs[k] > wf_max__max):
                    bad_pulses = np.append(bad_pulses, k)
                    continue
                    
            if wf_max__min is not None:
                if (pls_maxs[k] < wf_max__min):
                    bad_pulses = np.append(bad_pulses, k)
                    continue
        
        bad_pls_idxs[pulse_file] = bad_pulses
        print(pulse_file, ":", len(bad_pulses), "bad pulses")

    return bad_pls_idxs