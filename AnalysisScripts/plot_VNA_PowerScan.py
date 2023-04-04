from __future__ import division
import sys,os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm

import pandas as pd
from   glob import glob

sys.path.append('../BackendTools')
import ResonanceFitter as fitres
import ResonanceFitResult as fitclass

sys.path.append('/home/nexus-admin/NEXUS_RF/AcquisitionScripts')
from VNAMeas import *

## Set up matplotlib options for plots
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 12})
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
dfc = plt.rcParams['axes.prop_cycle'].by_key()['color']

## Flag to display plots
show_plots = False

## Series identifier
day    = '20220216'
time   = '123052'
series = day + '_' + time

## Path to VNA data
dataPath = '/data/PowerSweeps/VNA/'

## Create a place to store processed output
out_path = '/data/ProcessedOutputs/out_' + series

def parse_args():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Plot and Fit the data acquired in a VNA power scan')

    # Power scan optional arguments
    parser.add_argument('-d', type=str,
                        help='Date of data acquisition [YYYYMMDD]')
    parser.add_argument('-t', type=str,
                        help='Time of data acquisition [hhmmss]')
    parser.add_argument('-s', type=str,
                        help='A specific series identifier (typically [YYYYMMDD_hhmmss]). This supercedes any supplied Date or Time')
    
    # Data path optional arguments
    parser.add_argument('-p', type=str,
                        help='Top-level directory for saved VNA data')

    # Optional show plots switch
    parser.add_argument('--show', action='store_true',
                    help='If specified, plots are displayed in addition to being saved.')

    # Now read the arguments
    args = parser.parse_args()

    return args

def create_dirs():
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("Storing output at",out_path)
    return 0

def get_input_files(series_str):

    ## Define the series path from the series
    srPath = dataPath + series_str.split("_")[0] + '/' + series_str + '/'

    ## File string format
    fn_prefix = "Psweep_P"
    fn_suffix = "_" + series_str + ".h5"

    ## Find and sort the relevant directories in the series
    print("Searching for files in:", srPath)
    print(" with prefix:", fn_prefix)
    print(" and  suffix:", fn_suffix)
    vna_file_list = glob(srPath+fn_prefix+'*'+fn_suffix)
    vna_file_list.sort(key=os.path.getmtime)
    print("Using files:")
    for fname in vna_file_list:
        print("-",fname)
    return vna_file_list

def fit_single_file(file_name, show_plots=True):
    ## Open the h5 file for this power and extract the class
    sweep = decode_hdf5(file_name)
    sweep.show()

    ## Extract the RF power from the h5 file
    print("Extracting data for power:",sweep.power,"dBm")

    ## Parse the file, get a complex S21 and frequency in GHz
    f = sweep.frequencies / 1.0e9
    z = sweep.S21realvals + 1j*sweep.S21imagvals

    ## Create an instance of a file fit result class
    this_f_r = fitclass.SingleFileResult(file_name)
    this_f_r.power = sweep.power
    this_f_r.start_T = sweep.start_T
    this_f_r.final_T = sweep.final_T

    ## Fit this data file
    fr, Qr, Qc, Qi, fig = fitres.sweep_fit(f,z,this_f_r,start_f=f[0],stop_f=f[-1],show_plots=show_plots)

    if (len(fr) > 1):
        fr = fr[0]
        Qr = Qr[0]
        Qc = Qc[0]
        Qi = Qi[0]

    ## Show the results of the fit
    this_f_r.show_fit_results()

    ## Save the figure
    if show_plots:
        ax = fig.gca()
        ax.set_title("Power: "+str(sweep.power)+" dBm, Temperature: "+str(np.mean(sweep.start_T))+" mK")
        fig.savefig(os.path.join(out_path,"freq_fit_P"+str(sweep.power)+"dBm.png"), format='png')

    ## Return the fit parameters
    return sweep.power, fr, Qr, Qc, Qi, this_f_r

if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Read in the arguments
    day      = args.d if args.d is not None else day
    time     = args.t if args.t is not None else time
    series   = args.s if args.s is not None else day + '_' + time
    dataPath = args.p if args.p is not None else dataPath

    show_plots = args.show if args.show is not None else show_plots

    ## Create a place to store processed output
    out_path = '/data/ProcessedOutputs/out_' + series

    ## Define all the lists in which we'll store fit parameters
    ## Make these fixed size arrays or delete completely since it seems
    ## to be a little redundant
    fr_list = []; Qr_list = []; Qc_list = []; Qi_list = []; power_list = []
    
    ## Create somewhere for the output
    create_dirs()

    ## Get all the files for a specified series
    vna_files = get_input_files(series)

    ## Create a class instance containing the fit results for this series
    result = fitres.SeriesFitResult(day,series)
    result.resize_file_fits(len(vna_files))

    for i in np.arange(len(vna_files)):
        ## Fit this data file
        pwr, fr, Qr, Qc, Qi, res = fit_single_file(vna_files[i], show_plots=show_plots)
        result.file_fits[i] = res 
        result.powers[i] = pwr
        result.fit_fr[i] = fr
        result.fit_Qr[i] = Qr
        result.fit_Qi[i] = Qi
        result.fit_Qc[i] = Qc

        ## Store the fit results
        fr_list.append(fr); Qr_list.append(Qr)
        Qc_list.append(Qc); Qi_list.append(Qi)
        power_list.append(pwr)

    ## Store the fit results
    result.save_to_file(out_path)

    fig = plt.figure()
    plt.plot(power_list,fr_list)
    plt.xlabel('Applied RF Power [dBm]')
    plt.ylabel(r'Resonator frequency $f$ [Hz]')
    plt.title(r'Resonator frequency')
    fig.savefig(os.path.join(out_path,"f_vs_P.png"), format='png')

    fig = plt.figure()
    plt.plot(power_list,(np.mean(fr_list)-fr_list)/fr_list)
    plt.xlabel('Applied RF Power [dBm]')
    plt.ylabel(r'$\Delta f/f$')
    plt.title(r'Resonator frequency fractional shift')
    fig.savefig(os.path.join(out_path,"df_vs_P.png"), format='png')

    fig = plt.figure(figsize=(10,8))
    plt.plot(power_list,Qr_list)
    plt.xlabel('Applied RF Power [dBm]')
    plt.ylabel(r'Resonator Quality Factor $Q_r$')
    plt.title(r'Resonator Quality Factor $Q_r$')
    plt.tight_layout()
    fig.savefig(os.path.join(out_path,"Qr_vs_P.png"), format='png')

    fig = plt.figure(figsize=(10,8))
    plt.plot(power_list,Qc_list)
    plt.xlabel('Applied RF Power [dBm]')
    plt.ylabel(r'Coupling Quality Factor $Q_c$')
    plt.title(r'Coupling Quality Factor $Q_c$')
    plt.tight_layout()
    fig.savefig(os.path.join(out_path,"Qc_vs_P.png"), format='png')

    fig = plt.figure(figsize=(10,8))
    plt.plot(power_list,Qi_list)
    plt.xlabel('Applied RF Power [dBm]')
    plt.ylabel(r'Internal Quality Factor $Q_i$')
    plt.title(r'Internal Quality Factor $Q_i$')
    plt.tight_layout()
    fig.savefig(os.path.join(out_path,"Qi_vs_P.png"), format='png')

    if (show_plots):
        plt.show()