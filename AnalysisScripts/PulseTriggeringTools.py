import h5py

import numpy as np
import matplotlib.pyplot as plt

# from scipy.signal import periodogram,get_window,coherence
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
# from scipy.stats import skew

import TimestreamHelperFunctions as Thf


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
    res['Phase']=phases-np.angle(avg_S21s[0])
    res['Magnitude']=mags-abs(avg_S21s[0])
    res['Time']=times
    res['Fs']=1e6
    res['number_samples']=n_pts
    res['chan_names']=['Phase','Magnitude']
    res['cut_idx'] = np.argmin(np.abs(times-0.001))
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
 
    if (rising_edge): #rising edge
        if(verbose):
            print('Triggering on rising edge')
        trigA = (filtered_data[0:-1] < trig_th)
        trigB = (filtered_data[1:] > trig_th)
    else: #falling edge
        if(verbose):
            print('Triggering on falling edge')
        trigA = (filtered_data[0:1] > trig_th)
        trigB = (filtered_data[1:] < trig_th)
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
 
    res['trigpt'] = trigger_points
    res['series'] = np.full(n_trig,series)
    res['trigRate'] = np.full(n_trig,n_trig)
    res['trigPts'] = np.full(n_trig,n_trig_pts)
 
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
