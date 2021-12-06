#Generates fits for PT scans, plots parameters like f0 and q against power and temp
#for USRP .h5 files; theoretically easy to convert to CMT .txt files *with real and imaginary data*
#GS 2/21/12
from __future__ import division
import matplotlib
import cmath
#import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")
import sys,os
sys.path.append('/home/nexus-admin/workarea/PyMKID')
sys.path.append('/home/nexus-admin/workarea/GPU_SDR/scripts')
#from fit_VNA.py import *
import numpy as np
import argparse
from glob import glob
import datetime
try:
    import PyMKID_USRP_functions as puf
except ImportError:
    print("Cannot find PyMKID_USRP_functions")

#try:
#    import pyUSRP as u
#except ImportError:
#    try:
#        sys.path.append('/home/nexus-admin/workarea/GPU_SDR')
#        import pyUSRP as u
#    except ImportError:
#        print("Cannot find the pyUSRP package")

datapath='/data/TempSweeps/'
date='20210302'
time='213525'
#213525 for w/ filters, 152824 for w/out

dateStr=str(datetime.datetime.now().strftime('%Y%m%d_%H%M'))

powers = [-70,-65,-60,-55,-50,-45,-40,-35,-30]

Al_f0=[4.2425]
window_size=5000
#power=-60
#backend = 'matplotlib'
#attenuation=None
#smoothing=None
#a_cutoff=10
#threshold=None
#peak_width=20e3
#Mag_depth_cutoff=1
#N_peaks=2

Needs_Fitting_Old=False
Needs_Fitting=True
Plot_Fits_Old=False
Plot_Fits=False
Save_File_Old=False
#variables for sorting fits to their respective resonators
f0_centers = [4242.5, 4245.0] #in MHz, for Al then Nb
f0_delta = 0.5
for power in powers:
    vna_files = glob(datapath+date+'/*P'+str(power)+'*'+date+"_"+time+'.h5')
    vna_files.sort(key=os.path.getmtime)
    file_list=vna_files
    print("Fitting all traces at "+str(power)+" dBm")
    #if Needs_Fitting_Old:
        #for i in range(len(file_list)):
            #print(str(file_list[i])+"\n")
            #u.VNA_analysis(file_list[i])
            #if threshold is not None:
                #u.extimate_peak_number(file_list[i], threshold = threshold, smoothing = smoothing, peak_width = peak_width, verbose = False, exclude_center = True, diagnostic_plots = True)
            #else:
                #u.initialize_peaks(file_list[i], N_peaks = N_peaks[i], a_cutoff = a_cutoff, smoothing = smoothing, peak_width = peak_width, Qr_cutoff=4e3, verbose = False, exclude_center = True, diagnostic_plots = False,  Mag_depth_cutoff = Mag_depth_cutoff)
                #try:
                    #u.initialize_peaks(file_list[i], N_peaks = 2)
                #except UnboundLocalError:
                    #print("Could not initialize "+str(file_list[i]))
            #try:
                #u.vna_fit(file_list[i])
            #except ValueError:
                #print("Could not fit "+str(file_list[i]))
                #delete from filelist
                #del_name = file_list.pop(i)
#    if Plot_Fits_Old:
#        for i in range(len(file_list)):
#     	    #all resonator plotted on a single static png can be overwelming
#            #retrieve T value from filename
#            temp_val=''
#            offset=file_list[i].find("_T")
#            for j in range(3):
#                if (file_list[i][offset+2+j] != '_'):
#                    temp_val += file_list[i][offset+2+j]

    	    #if backend == 'matplotlib':
        	#single_plots = True
    	    #else:
#            single_plots = False
#            fname_reso='Reso_'+'T'+temp_val+'_P'+str(power)
#            fname_vna='VNA_'+'T'+temp_val+'_P'+str(power)
#            try:
#    	        u.plot_resonators(file_list[i], reso_freq = None, backend = backend, title_info = None, verbose = False, output_filename = fname_reso, auto_open = True, attenuation = None,single_plots = single_plots)
#    	        u.plot_VNA(file_list[i], backend = backend, output_filename=fname_vna)
#            except ValueError:
#                print("No resonators found for "+str(file_list[i]))
#                #delete from filelist
#                del_name = file_list.pop(i)

    if Save_File_Old:
        results = []
        for i in range(len(file_list)):
            #retrieve T value from filename
            temp_val=''
            offset=file_list[i].find("_T")
            for j in range(3):
                if (file_list[i][offset+2+j] != '_'):
                    temp_val += file_list[i][offset+2+j]
            #retrieve fit params
            for k in range(len(f0_centers)):
                try:
#                    fitvals=u.get_fit_param(file_list[i])
                    row = []
                    if (fitvals[k]["f0"]>(f0_centers[0]-f0_delta) and fitvals[k]["f0"]<(f0_centers[0]+f0_delta)):
                        row.append("Al")
                    elif (fitvals[k]["f0"]>(f0_centers[1]-f0_delta) and fitvals[k]["f0"]<(f0_centers[1]+f0_delta)):
                        row.append("Nb")
                    else:
                        row.append("?")
                        print("Res freq "+str(k)+" for temp "+str(temp_val)+" didn't fit any resonator")
                    row.append(float(temp_val))
                    row.append(fitvals[k]["f0"])
                    row.append(fitvals[k]["Qr"])
                    row.append(fitvals[k]["Qi"])
                    row.append(fitvals[k]["Qe"])
                    row.append(fitvals[k]["A"])
                    row.append(fitvals[k]["phi"])
                    row.append(fitvals[k]["a"])
                    row.append(fitvals[k]["D"])
                    results.append(row)

                    #print("Got fit vals for temp "+str(temp_val))
                except:
                    print("No fit params found for" + file_list[i]+"\n")

        #Sort into two separate arrays, one for each resonator
        al_results=[]
        nb_results=[]
        for row in results:
            if row[0] == "Al":
                al_results.append(row)
            elif row[0]=="Nb":
                nb_results.append(row)
            else:
                print("sorting failed at "+str(row[1])+"mK")

        dateStr=str(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
        savepath = datapath+date+'/fits/'+dateStr
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        #Create Al output file
        f = open(savepath+"/Al_P"+str(power)+'_'+date+".txt", "a")
        f.write("Temp(mK),f0(MHz),Qr,Qi,Qe,A,phi,a,D\n")
        for row in al_results:
            line = ''
            for i in range(8): 
                line += str(row[i+1])
                line += ","
            line += str(row[9])+"\n"
            f.write(line)

        f.close()

        #Create Nb output file
        f = open(savepath+"/Nb_P"+str(power)+'_'+date+".txt", "a")
        f.write("Temp(mK),f0(MHz),Qr,Qi,Qe,A,phi,a,D\n")
        for row in nb_results:
            line = ''
            for i in range(8):
                line += str(row[i+1])
                line += ","
            line += str(row[9])+"\n"
            f.write(line)

        f.close()

    if Needs_Fitting:
        results=[]
        for i in range(len(file_list)):
            #retrieve T value from filename
            temp_val=''
            offset=file_list[i].find("_T")
            for j in range(3):
                if (file_list[i][offset+2+j] != '_'):
                    temp_val += file_list[i][offset+2+j]
            #print("Fitting "+str(file_list[i]))
            fr, Qr, Qc_hat, a, phi, tau, Qc_real = puf.vna_file_fit(file_list[i],Al_f0,window_size,Plot_Fits)
            row = []
            row.append(float(temp_val))
            row.append(fr[0]*1e3)
            row.append(Qr[0])
            row.append(Qc_real)
            row.append(Qc_hat.real)
            row.append(Qc_hat.imag)
            row.append(phi)
            row.append(a.real)
            row.append(a.imag)
            row.append(tau.real)
            row.append(tau.imag)
            results.append(row)

        #dateStr=str(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
        savepath = datapath+date+'/PyMKID_fits/'+dateStr
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        #Create output file
        #hourStr=str(datetime.datetime.now().strftime('%Y%m%d_%H'))
        f = open(savepath+"/P"+str(power)+'_'+dateStr+".txt", "a")
        f.write("Temp(mK),f0(MHz),Qr,Qc_real,Qc_hat_real,Qc_hat_imag,phi,a_real,a_imag,tau_real,tau_imag\n")
        for row in results:
            line = ''
            for i in range(10):
                line += str(row[i])
                line += ","
            line += str(row[10])+"\n"
            f.write(line)

        f.close()
