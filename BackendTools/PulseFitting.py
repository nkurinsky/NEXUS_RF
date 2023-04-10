import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import expit

#### Fixed parameters for the fit equations ####
t0 = 4.99 ## ms

fixed_params = {"t0_ms": t0}

def set_fixed_param(key, value):
    fixed_params[key] = value
    return None

def get_fixed_param(key):
    return fixed_params[key]

def show_fixed_params():
    for k,v in fixed_params.items():
        print("\""+k+"\"",v)
    return None

#### Fitting equations #### 

def exp_fit(t,A,tau):
    t0 = get_fixed_param("t0_ms")
    return A*np.exp(-t/tau)*np.heaviside(t-t0,0.5)

def double_exp(t,a,tau,kappa):
    t0 = get_fixed_param("t0_ms")
    shape = a * (1 - expit(-(t-t0)/tau)) * expit(-(t-t0)/kappa) *np.heaviside(t-t0,0.5)
    return shape

def pulse_shape(t,Ad,Ap,tauP,kappaP):
    prompt = double_exp(t,Ap,tauP,kappaP)
    return prompt + Ad*delayed/np.max(delayed)

def dbl_pls_shape(t,aD,tD,kD,aP,tP,kP):
    t0 = get_fixed_param("t0_ms")
    prmpt = aP * (1 - expit(-(t-t0)/tP)) * expit(-(t-t0)/kP)
    delay = aD * (1 - expit(-(t-t0)/tD)) * expit(-(t-t0)/kD)
    return (prmpt+delay)*np.heaviside(t-t0,0.5)

#### Fit routine methods ####

## For a pulse timestream, estimate the amplitude and time constants
## Arguments
##   - t_vals         <array of float>     real time values in ms
##   - p_vals         <array of float>     real pulse timestream values
##   - t_cutoff_ms    <float>              time cutoff in ms, ignores timestream beyond this
##   - verbose        <bool or int>        flag for text output
## Returns
##   - param_est      <dict>               container of relevant/useful parameters
##       - amp        <float>              guess for amplitude of plain decaying exponential at t=0, inferred from data
##       - tau        <float>              guess for time constant of plain decaying exponential, inferred from data
##       - tmax       <float>              time of pulse maximum (in ms)
##       - pmax       <float>              value of pulse maximum 
def estimate_params(t_vals, p_vals, t_cutoff_ms=15.0, verbose=False):

    ## Trim the end of the timestream
    p_vals = p_vals[t_vals<t_cutoff_ms]
    t_vals = t_vals[t_vals<t_cutoff_ms]

    ## Make some guesses of the prompt parameters
    _p_max = np.max(p_vals)
    _t_max = t_vals[np.argmax(p_vals)]
    
    _ts_vals = t_vals[t_vals>t0][3:]
    _ps_vals = p_vals[t_vals>t0][3:]
    
    _p_1pe = _p_max*np.exp(-1)
    _t_1pe = _ts_vals[np.argmin(np.abs(_ps_vals-_p_1pe))]
    
    ## Turn those quantities into parameters for the prompt values
    tau_guess = _t_1pe - _t_max
    amp_guess = np.max(p_vals)*np.exp(t0/tau_guess)

    ## Ensure we found a positive lifetime
    tau_guess = np.max([0.001,tau_guess])
    
    ## Initialize our guess array for fitting to a single exponential
    if verbose > 0:
        print("-- Parameter guess for single exponential -- ")
        print("     Decay time constant: ", popt[1], "ms")
        print("     Amplitude at t=0:    ", amp_guess)

    param_est = {"amp": amp_guess,
                 "tau": tau_guess,
                 "tmax": _t_max  ,
                 "pmax": _p_max  }

    return param_est

## For a pulse timestream, estimate the amplitude and time constants
## Arguments
##   - t_vals          <array of float>    real time values in ms
##   - p_vals          <array of float>    real pulse timestream values
##   - param_est       <dict>              output of estimate_params()
##   - tp_guess        <float>             guess for time constant: prompt rise
##   - td_guess        <float>             guess for time constant: delayed rise
##   - kd_fac_guess    <float>             guess for relative factor between delayed decay and prompt decay time constants (factor in numerator)
##   - ad_fac_guess    <float>             guess for relative factor between delayed and prompt amplitudes (factor in denominator)
##   - t_cutoff_ms     <float>             time cutoff in ms, ignores timestream beyond this
## Returns
##   - param_guess     <array of float>    1D Ordered array of estimated parameters for pulse shape function: aD, tD, kD, aP, tP, kP
##   - param_guess     <array of float>    1D Ordered array of optimal parameters for pulse shape function: aD, tD, kD, aP, tP, kP
##   - param_guess     <array of float>    2D Covariance matrix of optimal parameters
def run_fit(t_vals, p_vals, param_est, tp_guess=0.01, td_guess=0.1, kd_fac_guess=10.0, ad_fac_guess=10.0, t_cutoff_ms=15.0):

    ## Extract the values from parameter estimation
    kp_guess = param_est["tau"]
    t_max    = param_est["tmax"]
    p_max    = param_est["pmax"]

    ## Define parameter guesses for the prompt component
    prompt_t = tp_guess
    prompt_a = p_max * np.exp((t_max-t0)/kp_guess) / (1-np.exp(-(t_max-t0)/prompt_t))
    prompt_k = kp_guess
    
    ## Define parameter guesses for the delayed component
    delay_a = prompt_a/ad_fac_guess
    delay_t = td_guess
    delay_k = kd_fac_guess*kp_guess
    
    ## Prepare to plot the prompt guess
    param_guess = [delay_a,delay_t,delay_k,prompt_a,prompt_t,prompt_k]

    ## Trim the end of the timestream
    p_vals = p_vals[t_vals<t_cutoff_ms]
    t_vals = t_vals[t_vals<t_cutoff_ms]
    
    ## Fit the overall shape
    param_opt, param_cov = curve_fit(dbl_pls_shape,t_vals,p_vals,p0=param_guess,bounds=(0,np.inf))

    return param_guess, param_opt, param_cov

# class PulseFitResult:

#     ## Rising edge of the pulse in milliseconds
#     t0 = -1.0

#     ## Parameter guess to initialize fit routine
#     p_est = {
#         "amp": -1.0,
#         "tau": -1.0,
#     }

#     ## Parameter optimal values after fit routine
#     p_opt = {
#         "amp": -1.0,
#         "tau": -1.0,
#     }

#     def __init__(self,t0):
#         ## ... 
#         self.t0 = t0
#         return None

#     def show_est_par(self):
#         print("Parameter initial estimate:")
#         for k,v in self.p_est.items():
#             print("\t",k,":",v)

#     def show_rough_result(self):
#         print("Parameter optimal values:")
#         for k,v in self.p_est.items():
#             print("\t",k,":",v)

#     def n_params(self):
#         return len(self.p_est)

#     def estimate_params(self,wf,t0,sr,bl_avg_samps=100):
#         ## Get the time and pulse value arrays
#         ## Here, we convert time to milliseconds explicitly
#         t_vals = np.arange(len(wf))/sr*1e3
#         p_vals = np.angle(wf)-np.mean(np.angle(wf)[0:int(bl_avg_samps)])
        
#         ## Locate the maximum of the pule
#         _p_max = np.max(p_vals)
#         _t_max = t_vals[np.argmax(p_vals)]

#         ## Find 1/e of its maximum value
#         _p_1pe = _p_max*np.exp(-1)
        
#         ## Create temporary arrays to look only after the rising edge
#         _ts_vals = t_vals[t_vals>t0][1:]
#         _ps_vals = p_vals[t_vals>t0][1:]
        
#         ## Find the time at which its fallen to 1/e of its max value
#         _t_1pe = _ts_vals[np.argmin(np.abs(_ps_vals-_p_1pe))]
        
#         ## Make your guess
#         tau_guess = _t_1pe - _t_max
#         amp_guess = np.max(p_vals)*np.exp(t0/tau_guess)

#         ## Save the guess
#         self.p_est["amp"] = amp_guess ## radians
#         self.p_est["tau"] = tau_guess ## milliseconds

#         return {
#             "amp": amp_guess, ## radians
#             "tau": tau_guess, ## milliseconds
#         }

#     def guess_dblexp_params(self,amp2,tau2):

#         ## Check that we have valid single exponential parameters
#         if self.p_est["amp"] < 0 and self.p_est["tau"]:
#             print("Single exponential parameters must be estimated first")
#             return None

#         ## Check if we've run this already
#         try:
#             self.p_est["amp2"]
#         except:
#             print("WARNING: You've already run this, which breaks the")
#             print("         iterative estimate of the new amplitude. ")
#             print("         Please re-estimate the single parameters.")

#         ## Save the guess
#         self.p_est["amp2"] = amp2 ## radians
#         self.p_est["tau2"] = tau2 ## milliseconds

#         ## Redefine the amplitude
#         new_amp = self.p_est["amp"]/(1+amp2*np.exp(t0*(1/self.p_est["tau"] - 1/tau2)))
#         self.p_est["amp"] = new_amp

#         return {
#             "amp" : new_amp,
#             "tau" : self.p_est["tau"],
#             "amp2": amp2,
#             "tau2": tau2,
#         }

