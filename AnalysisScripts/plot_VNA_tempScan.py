from __future__ import division
import sys,os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import cm

import pandas as pd
from   glob import glob

import ResonanceFitter as fitres
import ResonanceFitResult as fitclass

sys.path.append('/home/nexus-admin/NEXUS_RF/AcquisitionScripts')
from VNAMeas import *

import PyMKID_USRP_functions as puf

## Set up matplotlib options for plots
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 12})
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
dfc = plt.rcParams['axes.prop_cycle'].by_key()['color']

## Flag to display plots
show_plots = False

## Series identifier
day    = '20220419'
time   = '112746'
series = day + '_' + time

## Path to VNA data
dataPath = '/data/TempSweeps/VNA/'

## Create a place to store processed output
out_path = '/data/ProcessedOutputs/out_' + series

## Which power to look at
power='-50'
norm = plt.Normalize(vmin=10,vmax=350)

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

    ## Get all folders in date
    datePath    = os.path.join(dataPath, day)
    series_list = glob(datePath+"/"+day+"_*")

    ## Define the series path from the series
    srPath = os.path.join(datePath, day+"_*")

    ## File string format
    fn_prefix = "Psweep_P"
    fn_suffix = "_" + series_str + ".h5"

    ## Find and sort the relevant directories in the series
    print("Searching for files in:", srPath)
    print(" with prefix:", fn_prefix)
    print(" and  suffix:", fn_suffix)
    vna_file_list = glob(os.path.join(srPath,fn_prefix+str(power)+'*'+fn_suffix))
    vna_file_list.sort(key=os.path.getmtime)
    print("Using files:")
    for fname in vna_file_list:
        print("-",fname)
    return vna_file_list

def fit_single_file(file_name):
    raw_f, raw_VNA, amplitude = puf.read_vna(fname)
    #power = -14 + 20*np.log10(amplitude)

    temp = fname.split('_')[1][1:]
    color = cm.jet(norm(float(temp)))
    
    plt.figure(2,figsize=(8,6))
    plt.plot(raw_f,20*np.log10(abs(raw_VNA)),label=temp+' mK',color=color)

    # ## Open the h5 file for this power and extract the class
    # sweep = decode_hdf5(file_name)
    # sweep.show()

    # ## Extract the RF power from the h5 file
    # print("Extracting data for power:",sweep.power,"dBm")
    # power_list.append(sweep.power)

    # ## Parse the file, get a complex S21 and frequency in GHz
    # f = sweep.frequencies / 1.0e9
    # z = sweep.S21realvals + 1j*sweep.S21imagvals

    # ## Create an instance of a file fit result class
    # this_f_r = fitclass.SingleFileResult(file_name)
    # this_f_r.power = sweep.power
    # this_f_r.start_T = sweep.start_T
    # this_f_r.final_T = sweep.final_T

    # ## Fit this data file
    # fr, Qr, Qc, Qi, fig = fitres.sweep_fit(f,z,this_f_r,start_f=f[0],stop_f=f[-1])

    # if (len(fr) > 1):
    #     fr = fr[0]
    #     Qr = Qr[0]
    #     Qc = Qc[0]
    #     Qi = Qi[0]

    # ## Show the results of the fit
    # this_f_r.show_fit_results()

    # ## Save the figure
    # plt.gcf()
    # plt.title("Power: "+str(sweep.power)+" dBm, Temperature: "+str(np.mean(sweep.start_T))+" mK")
    # fig.savefig(os.path.join(out_path,"freq_fit_P"+str(sweep.power)+"dBm.png"), format='png')

    # ## Return the fit parameters
    # return sweep.power, fr, Qr, Qc, Qi, this_f_r

if __name__ == "__main__":

    ## Parse command line arguments to set parameters
    args = parse_args()

    ## Read in the arguments
    day      = args.d if args.d is not None else day
    time     = args.t if args.t is not None else time
    series   = args.s if args.s is not None else day + '_' + time
    dataPath = args.p if args.p is not None else dataPath

    show_plots = args.show if args.show is not None else show_plots
    
    ## Create somewhere for the output
    create_dirs()

    ## Get all the files for a specified series
    vna_files = get_input_files(series)

    ## Create a class instance containing the fit results for this series
    # result = fitres.SeriesFitResult(day,series)
    # result.resize_file_fits(len(vna_files))

    for i in np.arange(len(vna_files)):
        ## Fit this data file
        # pwr, fr, Qr, Qc, Qi, res = 
        fit_single_file(vna_files[i])
        # result.file_fits[i] = res 
        # result.powers[i] = pwr
        # result.fit_fr[i] = fr
        # result.fit_Qr[i] = Qr
        # result.fit_Qi[i] = Qc
        # result.fit_Qc[i] = Qi

        # ## Store the fit results
        # fr_list.append(fr); Qr_list.append(Qr)
        # Qc_list.append(Qc); Qi_list.append(Qi)

    ## Store the fit results
    # result.save_to_file(out_path)

    plt.title(('MKID Frequency Sweep at ' +power+' dBm'), fontdict = {'fontsize': 18})
    plt.figure(2)
    plt.xlabel('f [MHz]', fontdict = {'fontsize': 18})
    plt.ylabel('S21 [dB]', fontdict = {'fontsize': 18})
    cbar=plt.colorbar(cm.ScalarMappable(cmap=cm.jet, norm=norm),shrink=0.8)
    cbar.set_label('Temperature [mK]', size=16)
    #plt.legend(loc='center left',bbox_to_anchor=(1.,0.5))
    plt.tight_layout()

    if (show_plots):
        plt.show()




plt.show()













