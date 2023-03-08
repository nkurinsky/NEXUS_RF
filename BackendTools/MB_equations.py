import numpy as np
import iminuit
import scipy.special as spec

#############################
# iMinuit fitting functions #
#############################

######################
# Mattis-Bardeen fit #
######################

Boltz_k = 8.6173303E-5 # eV/K
N_0 = 1.72E28 # 1/(m^3*eV), Single-spin density of states (aluminum, from Jiansong's Thesis)
Planck_h = 4.135667662E-15 # eV*s

def signed_log10(x):
    return np.log10(np.abs(x)) * x/np.abs(x)

## Mazin thesis, equation 2.3: The density of thermally-excited quasiparticles
def n_qp(T, Delta0):
    # [K, eV]
    return 2.*N_0*np.sqrt(2.*np.pi*Boltz_k*T*Delta0)*np.exp(-1.*Delta0/(Boltz_k*T))

## Siegel thesis, equation 2.43
def kappa_1(T, f0, Delta0):
    xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
    # return (1/(np.pi*N_0))*np.sqrt(2./(np.pi*Boltz_k*T*Delta0))*np.sinh(xi)*spec.k0(xi)
    return (1/(np.pi*Delta0*N_0))*np.sqrt((2.*Delta0)/(np.pi*Boltz_k*T))*np.sinh(xi)*spec.k0(xi)

## Siegel thesis, equation 2.44
def kappa_2(T, f0, Delta0):
    xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
    return (1/(2.*Delta0*N_0))*(1.+np.sqrt((2.*Delta0)/(np.pi*Boltz_k*T))*np.exp(-1.*xi)*spec.i0(xi))

## Siegel thesis, equation 2.59
def f_T(T, f0, Delta0, alpha_f):
    # [K, Hz, eV, _]
    # xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
    # return -1.*alpha_f/(4.*Delta0*N_0) * ( 1. + np.sqrt((2.*Delta0)/(np.pi*Boltz_k*T)) * np.exp(-1.*xi) * spec.i0(xi) ) * n_qp(T,Delta0) * f0 + f0
    return f0 * (1 - 0.5 * alpha_f * kappa_2(T, f0, Delta0) * n_qp(T,Delta0))

## Siegel thesis, equation 2.60
def Qi_T(T, f0, Qi0, Delta0, alpha_Q):
    # xi = 1./2.*(Planck_h*f0)/(Boltz_k*T)
    # return ( alpha_Q/(np.pi*N_0) * np.sqrt(2./(np.pi*Boltz_k*T*Delta0)) * np.sinh(xi) * spec.k0(xi) * n_qp(T,Delta0) + 1./Qi0 )**-1.
    return 1./( alpha_Q * kappa_1(T, f0, Delta0) * n_qp(T,Delta0) + 1./Qi0)

def Qr_T(T, f0, Qi0, Delta0, alpha_Q):
    Qi = Qi_T(T, f0, Qi0, Delta0, alpha_Q)
    Qc = Qc_T(T, f0, Qi0, Delta0, alpha_Q)
    return (1./Qi) + (1./Qc)

## 2 * Delta * N0 * k1
def S_1(fr,T,Delta):
    # [Hz, K, eV]
    xi = 1./2.*(Planck_h*fr)/(Boltz_k*T)
    return (2/np.pi)*np.sqrt(2*Delta/(np.pi*Boltz_k*T))*np.sinh(xi)*spec.k0(xi) # unitless

## 2 * Delta * N0 * k2
def S_2(fr,T,Delta):
    # [Hz, K, eV]
    xi = 1./2.*(Planck_h*fr)/(Boltz_k*T)
    return 1+np.sqrt(2*Delta/(np.pi*Boltz_k*T))*np.exp(-1*xi)*spec.i0(xi) # unitless

## Fits to f + Qi, all parameters free
def MB_fitter(T_fit, Qi_fit, f_fit, fixed_alpha=False, fixed_delta=False, max_iters=500, verbose=False):

    ## Define the chi-squared expression
    def chisq(f0, Delta0, alpha, Qi0):
        ## First term in x^2 expression
        if Qi_fit is None:
            x2_t1  = 0
        else:
            var_Qi = np.var(Qi_fit)
            x2_t1  = (Qi_T(T_fit, f0, Qi0, Delta0, alpha) - Qi_fit)**2./var_Qi

        ## Second term in x^2 expression
        var_f = np.var(f_fit)
        x2_t2 = (f_T(T_fit, f0, Delta0, alpha) - f_fit)**2./var_f

        return sum( x2_t1 +  x2_t2 )

    ## Initialize parameters with a guess
    f0_in     = np.max(f_fit)  ## Hz
    Delta0_in = 0.17e-3   ## eV
    alpha_in  = 0.03801   ## frac
    Qi0_in    = Qi_fit[0] if Qi_fit is not None else -9999

    ## Do the minimization problem for 500 iterations
    for j in range(int(max_iters)):
        minimizer = iminuit.Minuit(chisq, 
            f0=f0_in, Delta0=Delta0_in, alpha=alpha_in, Qi0=Qi0_in, 
            limit_f0     = (np.max(f_fit)*0.75,np.max(f_fit)*1.25), 
            limit_Delta0 = (Delta0_in,Delta0_in) if fixed_delta else (1.0e-5,2.5e-4), 
            limit_alpha  = (alpha_in ,alpha_in ) if fixed_alpha else (1.0e-4,5.0e-2), 
            limit_Qi0    = (-9999    ,-9999 ) if Qi_fit is None else (1.e2,1.e7), 
            pedantic=False, print_level=-1 if not verbose else 0)

        f0_in     = minimizer.values["f0"]
        Delta0_in = minimizer.values["Delta0"]
        alpha_in  = minimizer.values["alpha"]
        Qi0_in    = minimizer.values["Qi0"]

        minimizer.migrad()

    ## Extract the final values from the minimization problem
    f0     = minimizer.values["f0"]
    Delta0 = minimizer.values["Delta0"]
    alpha  = minimizer.values["alpha"]
    Qi0    = minimizer.values["Qi0"]

    ## Get the degrees of freedom and reduced chisq
    ndof   = 4.0
    if (fixed_alpha):
        ndof -= 1.0
    if (fixed_delta):
        ndof -= 1.0
    if (Qi_fit is None):
        ndof -= 1.0

    chi_sq_dof = chisq(f0, Delta0, alpha, Qi0)/ndof

    ## F(T=0) [GHz] ; Delta(T=0) [meV] ; alpha(T=0) [frac.] ; Qi(T=0) ; reduced x2
    return f0/1e9, Delta0*1e3, alpha, Qi0, chi_sq_dof

## Fits to Qr rather than Qi
def MB_fitter_Qr(T_fit, Qr_fit, f_fit, fixed_alpha=False, fixed_delta=False, max_iters=500, verbose=False):
    ## Define the chi-squared expression
    def chisq(f0, Delta0, alpha, Qr0):
        ## First term in x^2 expression
        if Qr_fit is None:
            x2_t1  = 0
        else:
            var_Qr = np.var(Qr_fit)
            x2_t1  = (Qr_T(T_fit, f0, Qr0, Delta0, alpha) - Qr_fit)**2./var_Qr

        ## Second term in x^2 expression
        var_f = np.var(f_fit)
        x2_t2 = (f_T(T_fit, f0, Delta0, alpha) - f_fit)**2./var_f

        return sum( x2_t1 +  x2_t2 )

    ## Initialize parameters with a guess
    f0_in     = np.max(f_fit)  ## Hz
    Delta0_in = 0.17e-3   ## eV
    alpha_in  = 0.03801   ## frac
    Qr0_in    = Qr_fit[0] if Qr_fit is not None else -9999

    ## Do the minimization problem for 500 iterations
    for j in range(int(max_iters)):
        minimizer = iminuit.Minuit(chisq, 
            f0=f0_in, Delta0=Delta0_in, alpha=alpha_in, Qr0=Qr0_in, 
            limit_f0     = (np.max(f_fit)*0.75,np.max(f_fit)*1.25), 
            limit_Delta0 = (Delta0_in,Delta0_in) if fixed_delta else (1.0e-5,2.5e-4), 
            limit_alpha  = (alpha_in ,alpha_in ) if fixed_alpha else (1.0e-4,5.0e-2), 
            limit_Qr0    = (-9999    ,-9999 ) if Qr_fit is None else (1.e2,1.e7), 
            pedantic=False, print_level=-1 if not verbose else 0)

        f0_in     = minimizer.values["f0"]
        Delta0_in = minimizer.values["Delta0"]
        alpha_in  = minimizer.values["alpha"]
        Qr0_in    = minimizer.values["Qr0"]

        minimizer.migrad()

    ## Extract the final values from the minimization problem
    f0     = minimizer.values["f0"]
    Delta0 = minimizer.values["Delta0"]
    alpha  = minimizer.values["alpha"]
    Qr0    = minimizer.values["Qr0"]

    ## Get the degrees of freedom and reduced chisq
    ndof   = 4.0
    if (fixed_alpha):
        ndof -= 1.0
    if (fixed_delta):
        ndof -= 1.0
    if (Qr_fit is None):
        ndof -= 1.0

    chi_sq_dof = chisq(f0, Delta0, alpha, Qr0)/ndof

    ## F(T=0) [GHz] ; Delta(T=0) [meV] ; alpha(T=0) [frac.] ; Qr(T=0) ; reduced x2
    return f0/1e9, Delta0*1e3, alpha, Qr0, chi_sq_dof

#appends caltech data for f0 to higher temp vals. doesn't really work, too much of a jump
def MB_fitter2(T_fit, Qi_fit, f_fit,added_points=11):
    fit_result = []
    added_points=10
    def chisq_f(f0, Delta0, alpha):
        
        alpha_f = alpha

        var_f = np.var(f_fit)

        #return sum( (Qi_T(T_fit[0:(len(T_fit)-added_points)], f0, Qi0, Delta0, alpha_Q) - Qi_fit)**2./var_Qi + (f_T(T_fit, f0, Delta0, alpha_f) - f_fit)**2./var_f )
        return sum((f_T(T_fit, f0, Delta0, alpha) - f_fit)**2./var_f )

    def fit_chisq_test(T_fit, f_fit, Qi_fit, f0, Delta0, alpha, Qi0):
        var_Qi = np.var(Qi_fit)
        var_f = np.var(f_fit)

        return sum(( (Qi_T(T_fit[0:(len(T_fit)-added_points)], f0, Qi0, Delta0, alpha) - Qi_fit)**2./var_Qi) +sum((f_T(T_fit, f0, Delta0, alpha) - f_fit)**2./var_f))/4.

    f0_in = f_fit[0]
    Delta0_in = 4.e-4
    alpha_in = 0.03801
    #Qi0_in = Qi_fit[0]

    for j in range(100):
        minimizer = iminuit.Minuit(chisq_f, f0=f0_in, Delta0=Delta0_in, alpha=alpha_in, limit_f0=(f_fit[0]/1.1,f_fit[0]*1.1), limit_Delta0=(1.e-4,1.e-3), limit_alpha=(0.,0.5), pedantic=False, print_level=-1)

        f0_in = minimizer.values["f0"]
        Delta0_in = minimizer.values["Delta0"]
        alpha_in = minimizer.values["alpha"]
        #Qi0_in =minimizer.values["Qi0"]

        minimizer.migrad()

    f0 = minimizer.values["f0"]
    Delta0 = minimizer.values["Delta0"]
    alpha = minimizer.values["alpha"]
    #Qi0 =minimizer.values["Qi0"]

    def chisq_qi(Qi0):
        var_qi=np.var(Qi_fit)
        return sum((Qi_T(T_fit[0:(len(T_fit)-added_points)], f0, Qi0, Delta0, alpha) - Qi_fit)**2./var_qi )

    Qi0_in = Qi_fit[0]

    for j in range(100):
        minimizer = iminuit.Minuit(chisq_qi, Qi0=Qi0_in, limit_Qi0=(1e2,1e7), pedantic=False, print_level=-1)

        Qi0_in =minimizer.values["Qi0"]

        minimizer.migrad()

    Qi0=minimizer.values["Qi0"]
    chi_sq_dof = fit_chisq_test(T_fit, f_fit, Qi_fit, f0, Delta0, alpha, Qi0)

    fit_result.append([f0/1.e9,Delta0*1000,alpha,Qi0,chi_sq_dof])

    T_smooth = np.linspace(T_fit[0],T_fit[-1],10000)

    return f0/1.e9, Delta0*1000., alpha, Qi0, chi_sq_dof
