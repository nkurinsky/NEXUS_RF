import h5py

import numpy as np
import matplotlib.pyplot as plt

# from scipy.signal import periodogram,get_window,coherence
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
# from scipy.stats import skew

import TimestreamHelperFunctions as Thf

def readLegacyDataFile(filestr):
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

def readDataFile(series):
    sum_file, dly_file, vna_file, tone_files = Thf.GetFiles(series, verbose=True)

    try:
        metadata, avg_frqs, avg_S21s = Thf.UnpackSummary(sum_file)
    except:
        metadata = {'rate':1e6}
        avg_frqs = np.array([0])
        avg_S21s = np.array([0])
    
    f_tone = h5py.File(tone_files[0], 'r')
    
    res_ts = np.array(f_tone['raw_data0']['A_RX2']['data'])[0,:]
    n_pts  = len(res_ts)
    
    # times  = np.arange(n_pts)/1e6
    times  = np.arange(n_pts)/metadata['rate'] * 100
    mags   = abs(res_ts)
    phases = np.angle(res_ts)
    
    res = dict()
    res['Phase']     = phases#-np.angle(avg_S21s[0])
    res['Magnitude'] = mags#-abs(avg_S21s[0])
    res['Time']      = times
    res['Fs']        = 1e6
    res['number_samples']=n_pts
    res['chan_names']= ['Phase','Magnitude']
    res['cut_idx']   = np.argmin(np.abs(times-0.001))
    res['avg_mag']   = np.angle(avg_S21s[0])
    res['avg_phs']   = abs(avg_S21s[0])
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

def getEvents(series, trig_channel='Phase', trig_th = 2.0, rising_edge = True, maxAlign=True,
              pretrig = 1024, trace_len = 4096, trig_sep = 4096, ds=4, pretrig_template = 1024, 
              tauRise = 20e-6, tauFall = 300e-6, ACcoupled=True, verbose=False, template=None):
    '''
    This function takes data from continuous DAQ, where each file contains 1 sec of data
    It slices the 1sec of data into events with a level trigger
    
    Double exponential template is generated if no template is given. For OF template, run 'pulseFromTemplate' to generate
    apropriate template for use with trigger function
    
    '''
    res=readDataFile(series)
    chan_names=res['chan_names']
    number_samples=res['number_samples']
 
    #make sure trigger channel is valid
    chan_names = res['chan_names']
    if(trig_channel not in chan_names):
        trig_channel='Phase'
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
    filtered_data = fftconvolve(meanTrace, meanTemplate[::-1], mode="valid")
    #filtered_data = np.correlate(meanTrace,meanTemplate)*meandt
    # plt.plot(filtered_data)
    # plt.show()

    convolution_mean = np.mean(filtered_data)
    trigger_std      = trig_th*np.std(filtered_data)
 
    if (rising_edge): #rising edge
        if(verbose):
            print('Triggering on rising edge')
        trigA = (filtered_data[0:-1] < trig_th)
        trigB = (filtered_data[1:] > trig_th)
        # trigA = (filtered_data[0:-1] < convolution_mean + trigger_std)
        # trigB = (filtered_data[1:] > convolution_mean + trigger_std)
    else: #falling edge
        if(verbose):
            print('Triggering on falling edge')
        trigA = (filtered_data[0:1] > trig_th)
        trigB = (filtered_data[1:] < trig_th)
        # trigA = (filtered_data[0:1] > trigger_std)
        # trigB = (filtered_data[1:] < trigger_std)
    trigger_condition = trigA & trigB
    trigger_points=np.flatnonzero(trigger_condition)+1
 
    rm_index = []
    n_trig = len(trigger_points)
    n_trig_pts = n_trig
    idx = 0
    alignPreTrig = 200
    alignPostTrig =500
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
 
    res['trigpt']   = trigger_points
    res['trigtime'] = res['Time'][trigger_points]
    res['series']   = np.full(n_trig,series)
    res['trigRate'] = np.full(n_trig,n_trig)
    res['trigPts']  = np.full(n_trig,n_trig_pts)
 
    del singleTrace
    del filtered_data
    del trigger_condition
    del trigA,trigB
 
    return res

def movavg(y,side_pts=3):
    n_pts = 1 + 2*side_pts
    # y_avg = np.array([ np.sum(y[i-side_pts:i+side_pts])/n_pts for i in  side_pts+np.arange(lgth-n_pts)])
    y_avg = np.convolve(y, np.ones(n_pts), 'valid') / n_pts
    return y_avg

def movavg_xy(x,y,side_pts=3):
    
    lgth = 0
    if not (len(x) == len(y)):
        print("x and y must be the same size")
        return None
    else:
        lgth = len(x)
        
    
    if (lgth % 2 == 0 ):
        x = x[:-1]
        y = y[:-1]
    
    if not (type(x) == type(np.array([0]))):
        x = np.array(x)
    
    x_pts = x[side_pts:-side_pts]
    y_avg = movavg(y,side_pts)
    
    return x_pts, y_avg

def GetResponse(series, trig_channel="Phase", traceLength=4096, trig_th=1.0e4, 
                tauFall=500e-6, mean_pre_samps=800, doAlign=True, movAvgPts=50,
                verbose=False, show_plots=False):
    
    ## Initialize the pulse trackers
    pulseCount=0
    traces = {}

    ## Get the events for the specified trigger channel
    events=getEvents(series,ds=1,trig_th=trig_th, trace_len = traceLength, 
        trig_sep = traceLength, tauFall=tauFall)
    nEvents=len(events[trig_channel])
    for i in range(0,nEvents):
        pulseCount+=1
        trace=events[trig_channel][i]
        trace -= np.mean(trace[0:int(mean_pre_samps)])
        traces[pulseCount] = trace

    if verbose:
        print("Pulse count:", pulseCount)

    _tint = np.arange(-250,traceLength-1750)/1e3
    avg_trace = np.zeros(len(_tint))
    n_traces = 0

    if show_plots:
        plt.figure()

    for pulseNum in traces:
        if (pulseNum < 100) or True:
            
            trace = traces[pulseNum]
            
            ## Line up the traces
            pos_max = np.argmax(trace) if doAlign else 0
            tidxs = np.arange(0,traceLength)
            tvals = (tidxs-pos_max)/1e3
            
            if doAlign and (pos_max<500 or pos_max>1500):
                continue
            
            _intp = interp1d(tvals,trace,bounds_error=False,fill_value=0)
            _trin = _intp(_tint)
            
            avg_trace += _intp(_tint)
            n_traces  += 1
            
            if show_plots:
                av_t, av_w = movavg_xy(tvals,trace,side_pts=movAvgPts)
                plt.plot(av_t, av_w,alpha=0.25)
                # plt.plot(tvals,trace,alpha=0.25)

    if show_plots:    
        avg_trace /= n_traces
        plt.plot(_tint,avg_trace,color='k')

        # plt.ylabel('magnitude')
        plt.ylabel('phase (radians)')
        plt.xlabel('milliseconds')
        # plt.title('Examples of detected phonon pulses')
        plt.title(trig_channel+' Response')
        # plt.ylim([-0.30,0.30])
        # plt.ylim([-0.20,0.40])
        # plt.ylim([-0.60,0.60])

    return pulseCount, traces

def CalcPulseParams(traces, movAvgPts=None):
    pulse_heights = []
    taus = []
    # interestingPulses = []
    # interestingPulseHeights = []
    for pulseNum in traces:
        trace = traces[pulseNum]
        if movAvgPts is not None:
        	trace = movavg(trace,side_pts=movAvgPts)
        pulse_max = np.amax(trace[800:1800])
        pulse_heights.append(pulse_max)
        pulse_max_idxs = np.argwhere(trace == pulse_max)
        pulse_max_idx = pulse_max_idxs[0][0]
        trace_after_pulse = trace[pulse_max_idx:-500]
        tau = np.argmin(np.abs(trace_after_pulse - pulse_max/np.e))
        taus.append(tau)

        # if tau > 600 and tau < 800:
        #     interestingPulses.append(pulseNum)
        #     interestingPulseHeights.append(pulse_max)

    return pulse_heights, taus

## This function is used to stack pulses only based upon the location
## of the first pulse and the rate of pulse arrival. Used for finding 
## the average pulse to create a template for triggering in timestreams
## with irregular arrival times. The wall time values are inferred 
## from the total number of samples and sampling rate.
## The pulse rate determines the fixed amount of time and number of
## samples between KID response pulse rising edges. A window spanning
## the user-specified start time and the time between pulses is defined
## by a fraction. That window jumps along the waveform with the step
## set by the time between pulse arrival. The waveform in each window
## is stacked, then averaged.
##  
## INPUTS
## -- timestream      1D numpy array    Ordered values of a tone timestream
##                    of float          phase or magnitude in which to find
##                                      pulses
## -- start_t_sec     float             Point in time (seconds since start)
##                                      where pulses should start to be found
##                                      it should be a few tens of samples 
##                                      before the first rising edge of a 
##                                      pulse in the timestream (units: seconds)
## -- pulse_Hz_rate   float/int         [default 100] The rate of arrival of
##                                      pulses, usually set by the AWG driving
##                                      the LED or laser supplying the optical
##                                      photons (units: Hz)
## -- win_fac         float             [default 0.90] sets the window size as 
##                                      a fraction of the time between pulses
## -- sample_rate     float             [default 1e6] sample rate of the data, 
##                                      used to infer real time 
##                                      (units: sample per second)
## -- Npulses         int               [default None] total number of pulse 
##                                      windows to average - this is needed
##                                      if there is a pulse-free region at the 
##                                      end of the timestream, after the LED
##                                      exposure
## -- bl_subtract     bool              [default False] subtract off the mean
##                                      of the 2nd to 6th window in the pre
##                                      pulse timestream
## -- show_plots      bool              [default False] create a plot of the  
##                                      windows and the average waveform
## OUTPUTS
## -- avg_wvfm        1D numpy array    Ordered values of an average pulse with
##                    of float          the same length and sampling rate as 
##                                      the input array
## -- i               int               Number of pulse windows averaged
## -- baseline        float             The baseline that is found and removed
## -- window          int               Width of the window in samples
def StackPulses(timestream, start_t_sec, pulse_rate_Hz=100, win_fac=0.90, sample_rate=1e6, Npulses=None, 
                bl_subtract=False, show_plots=False, plot_time=False):
    ## Convert times to samples
    start_samp     = int(sample_rate * start_t_sec)
    t_btwn_pulses  = 1./pulse_rate_Hz
    samps_btwn_pls = int(sample_rate * t_btwn_pulses)
    
    ## Define the window of interest
    if win_fac > 0.95:
        win_fac = 0.95
    window = int(win_fac * samps_btwn_pls)
    
    ## Take the average of the second 5 windows in the timestream to determine a baseline
    baseline = np.mean(timestream[window:6*window])
    
    ## Subtract off the baseline from the waveform
    waveform = timestream 
    if bl_subtract:
        waveform-= baseline
    
    ## Create a storage container for the averaged waveform
    avg_wvfm = np.zeros(window)
    
    ## Initialize the plot
    if show_plots:
        fig = plt.figure()
        ax0 = fig.gca()
        
    ## Start an index counter
    i = 0

    ## Loop over a fixed number of samples starting at the user-input start time
    ## until the end of the window is beyond the end of the timestream
    while start_samp+i*samps_btwn_pls+window < len(timestream)-1:

        ## Add the values of the waveform in the current window to the averaged waveform
        avg_wvfm +=  waveform[start_samp+i*samps_btwn_pls:start_samp+i*samps_btwn_pls+window]
        
        ## Add the waveform in the current window to the plot
        if show_plots:
            if plot_time:
                ax0.plot(np.arange(window)/sample_rate,waveform[start_samp+i*samps_btwn_pls:start_samp+i*samps_btwn_pls+window], 
                    alpha=0.2)
            else:
                ax0.plot(waveform[start_samp+i*samps_btwn_pls:start_samp+i*samps_btwn_pls+window], 
                    alpha=0.2)

        ## Increment the index counter
        i+=1
        
        ## Stop after a certain number of pulses
        if Npulses is not None:
            if i>Npulses:
                break

    ## Average the summed waveform
    avg_wvfm/=i

    ## Draw the average waveform
    if show_plots:
        if plot_time:
            ax0.plot(np.arange(len(avg_wvfm))/sample_rate,avg_wvfm,"k--")
        else:
            ax0.plot(avg_wvfm,"k--")
        
    ## Return the averaged waveform
    return avg_wvfm, i, baseline, window    

def PlotPulse(timestream, start_t_sec, p_index=0, fig_obj=None,
              pulse_rate_Hz=100, win_fac=0.90, sample_rate=1e6,
              baseline=None, complx=False, plot_time=False):

    ## Convert times to samples
    start_samp     = int(sample_rate * start_t_sec)
    t_btwn_pulses  = 1./pulse_rate_Hz
    samps_btwn_pls = int(sample_rate * t_btwn_pulses)

    ## Define the window of interest
    if win_fac > 0.95:
        win_fac = 0.95
    window = int(win_fac * samps_btwn_pls)

    ## Determine sample index for the start of the specified window
    s_samp = start_samp+p_index*samps_btwn_pls

    ## Subtract the baseline if provided
    bl  = 0 if baseline is None else baseline
    wf  = timestream[s_samp:s_samp+window] - bl

    ## Create a plot object
    fig = plt.figure() if fig_obj is None else fig_obj
    ax0 = fig.gca()

    ## Draw the plot
    if complx:
        ax0.scatter(np.real(wf),np.imag(wf),alpha=0.2)
    else:
        if plot_time:
            ax0.plot(np.arange(len(wf))/sample_rate,wf)
        else:
            ax0.plot(wf)

    return wf

    








    
