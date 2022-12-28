#Utilities like plottinog and fitting USRP scan fit analysis
#plots are separated to here because pyUSRP doesn't like coexisting with local matplotlib use
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys,os
import math
import csv
import cmath #complex operations on Qe (Qc)
try:
    import MB_equations
except ImportError:
    try:
        sys.path.append('../BackendTools')
        import MB_equations
    except ImportError:
        print("Cannot find the MB_equations package")

def read_data_PyMKID(fname,num_quantities=11):
    #Default format: temp(mK),f0(MHz),Qr,Qc_real,Qc_hat_real,Qc_hat_imag,phi,a_real,a_imag,tau_real,tau_imag 
    f = open(fname, "r")

    #remove header
    buffer = f.readline()

    results = []
    line = f.readline()
    while len(line)>2:
        line = line.rstrip("\n")
        data = line.split(",")
        if len(data) != num_quantities:
            print("File had unexpected format\n")
            return []

        #convert the data from string format to appropriate format
        for i in range(num_quantities):
            try:
                data[i]=float(data[i])
            except:
                data[i]=0
                print("Data for column "+str(i)+" (starting at 0) could not be float converted")

        results.append(data)
        line = f.readline()

    f.close()
    #Sort rows of results array by value of column 0 (temperature)
    results=np.array(results)
    sorted_results=results[np.argsort(results[:,0])]
    return sorted_results

def read_data_PyUSRP(fname,num_quantities=9):
    #num_quantities is 9 for modified PyUSRP h5 fit output: temp, f0, Qr, Qi, Qe, A, phi, a, D
    #By necessity, the results array generated here has 10 columns: temp, f0, Qr, Qi, Qe_real, A, phi, a, D, Qe_imag
    f = open(fname, "r")

    #remove header
    buffer = f.readline()

    results = []
    line = f.readline()
    while len(line)>2:
        line = line.rstrip("\n")
        data = line.split(",")
        if len(data) != num_quantities:
            print("File had unexpected format\n")
            return []

        #convert the data from string format to appropriate format
        for i in range(9):
            if i == 3:
                if data[i]=="None":
                    data[i]=0
                else:
                    data[i]=float(data[i])
            if i == 4:
                data[i]=complex(data[i])
                data.append(data[i].imag)
                data[i] = data[i].real
            else:
                try:
                    data[i]=float(data[i])
                except:
                    data[i]=0
                    print("Data for column "+str(i)+" (starting at 0) could not be float converted")

        results.append(data)
        line = f.readline()

    f.close()
    return np.array(results)


def round_temps(temps):
    newtemps = []
    for temp in temps:
        newtemp = 5*round(temp/5)
        newtemps.append(newtemp)
    return newtemps

def plot_df0_f0(results,power,resonator="Al"):
    plt.figure(1,figsize=(10,10))
    plt.plot(results[:,0],results[:,10],'o')
    plt.title("df0/f0 vs T of " + resonator + "resonator at " + str(power)+' dBm')
    plt.figure(1)
    plt.xlabel('T (mK)')
    plt.ylabel('df0/f0')
    plt.show()
    return

def plot_qr(results,power,resonator="Al"):
    plt.figure(2,figsize=(10,10))
    plt.plot(results[:,0],(results[:,2]/(1e6)),'o')
    plt.title("Qr vs T of "+resonator +"resonator at " + str(power)+' dBm')
    plt.figure(2)
    plt.xlabel('T (mK)')
    plt.ylabel('Qr x 10^6')
    plt.show()
    return

def delete_outliers(results,outlier_temps):
    kill_list = []
    for temp in outlier_temps:
        for i in range(len(results[:,0])):
            if abs(results[i,0]-temp) < 1e-6:
                kill_list.append(i)

    if len(kill_list) != 0:
        reduced_results = np.delete(results, kill_list, 0) #0 is axis of deletion, in this case rows corresponding to a scan at a temp
    else:
        reduced_results = results
    return reduced_results

def fit_fits(results,  append_ctdata, c_temps, c_fs):
    #Need to be in units of K,__, and Hz. Convert from mK and MHz
    if not append_ctdata:
        fit_results = MB_equations.MB_fitter4(results[:,0]/1e3,results[:,11],results[:,1]*1e6)
    else:
        startnum=19
        temps=np.append(results[:,0],c_temps[startnum:(startnum+10)])
        print(temps)
        fs=np.append(results[:,1],c_fs[startnum:(startnum+10)])
        print(fs)
        fit_results = MB_equations.MB_fitter2(temps/1e3,results[:,11],fs*1e6)
    #print(fit_results)
    return list(fit_results)

def display_MBfitresults(results,label1, append_ctdata=False, c_temps=[], c_fs=[]):
    fr=fit_fits(results, append_ctdata, c_temps, c_fs)
    fr[2]=fr[2]*100 #convert alpha to percent
    dfr=[]#display fit results
    for i in range(len(fr)):
        val = str(round(fr[i],(6-int(math.floor(math.log10(abs(fr[i])))) - 1)))#first int is how many sig figs you want displayed
        dfr.append(val)
    print("Fit results for "+label1+" are f0 = "+dfr[0]+" Ghz, Delta0 = "+dfr[1]+" meV, alpha is "+dfr[2]+"%, Qi0 is "+dfr[3]+", Chi-squared-dof is "+dfr[4])
    fr[2]=fr[2]/100 #convert alpha back from percent
    return fr, dfr

def compare_plots_qi(results, results2, label1, label2, fr1, fr2, show_MB=True):
    plt.figure(1,figsize=(8,8))
    plt.plot(results[:,0],(results[:,11]/1e6),'ob',label = label1)
    plt.plot(results2[:,0],(results2[:,11]/1e6),'vr',label = label2)
    if show_MB:
        MB_qi1=calc_MB_theory(results, fr1,'qi')
        MB_qi2=calc_MB_theory(results2, fr2,'qi')
        plt.plot(results[:,0],MB_qi1*1e-6,'-b',label=(label1+' fit'))
        plt.plot(results2[:,0],MB_qi2*1e-6,'-r',label=(label2+' fit'))
    plt.title("Qi vs T of " + resonator + " resonator at " + str(power)+' dBm showing filter effect')
    plt.xlabel('T (mK)')
    plt.ylabel('Internal Quality Factor Qi (1e6)')
    plt.legend()
    plt.show()
    return

def masterplot(results, results2, label1, label2, fr1, fr2, dfr1, dfr2, show_MB=True, show_ct=True, ct_temps=[], ct_df_f0s=[], ct_qis=[], ct_MB_df_f0s=[], ct_MB_qis=[], ct_dfr=[]):
    #caltech relevant appending range: 19:29
    if show_MB:
        MB_qi1=calc_MB_theory(results, fr1,'qi')
        MB_qi2=calc_MB_theory(results2, fr2,'qi')
        MB_df_f01=calc_MB_theory(results, fr1,'df/f0')
        MB_df_f02=calc_MB_theory(results2, fr2,'df/f0')
    plt.figure(2,figsize=(12,6))
    #matplotlib.rcParams['legend.loc']='lower left'
    #df_f0 plot
    plt.subplot(1,2,1)
    plt.plot(results[:,0],(results[:,12]),'ob',label = label1)
    plt.plot(results2[:,0],(results2[:,12]),'vr',label = label2)
    #Delta0 results: +r'$\Delta$ = '+dfr1[1]+' meV\n'
    fitlabel1=label1+' Fit:\n'+r'f$_0$ = '+ dfr1[0]+' Ghz\n'+r'Q$_0$ = '+dfr1[3]+'\n'+r'$\alpha = $'+dfr1[2]+'%'
    fitlabel2=label2+' Fit:\n'+r'f$_0$ = '+ dfr2[0]+' Ghz\n'+r'Q$_0$ = '+dfr2[3]+'\n'+r'$\alpha = $'+dfr2[2]+'%'
    plt.plot(results[:,0],MB_df_f01,'-b',label=(fitlabel1))
    plt.plot(results2[:,0],MB_df_f02,'-r',label=(fitlabel2))
    if (len(ct_temps)!=0) and (show_ct):
        plt.plot(ct_temps[0:21], ct_df_f0s[0:21], 'ok', label = "Caltech Data")
        plt.plot(ct_temps[0:21], ct_MB_df_f0s[0:21], '-k', label = "Caltech MB Fits")
    plt.title("df0/f0 vs T, " + str(power)+' dBm')
    plt.xlabel('T (mK)')
    plt.ylabel('df0/f0')
    plt.legend()
    

    #Qi plot
    plt.subplot(1,2,2)
    plt.plot(results[:,0],(results[:,11]/1e6),'ob',label = label1)
    plt.plot(results2[:,0],(results2[:,11]/1e6),'vr',label = label2)
    #Construct fit parameter label
    #label1+" fit:\n$f_0$ = "+dfr1[0]+" Ghz\n$^\Delta_0 = $"+dfr1[1]+" meV\n$^\alpha = $"+dfr1[2]+"%\nQi0 = "+dfr1[3]
    
   
    plt.plot(results[:,0],MB_qi1*1e-6,'-b',label=label1 + " Fit")
    plt.plot(results2[:,0],MB_qi2*1e-6,'-r',label=label2+" Fit")
    if (len(ct_temps)!=0) and (show_ct):
        #ct_fitlabel=label2+' fit:\n'+r'f$_0$ = '+ ct_dfr[0]+' Ghz\n'+r'Q$_0$ = '+ct_dfr[3]+'\n'+r'$\Delta$ = '+ct_dfr[1]+' meV\n'+r'$\alpha = $'+ct_dfr[2]+'%'
        ct_fitlabel="Caltech MB Fit"
        plt.plot(ct_temps[0:21], np.array(ct_qis[0:21])*1e-6, 'ok', label = "Caltech Data")
        plt.plot(ct_temps[0:21], np.array(ct_MB_qis[0:21])*1e-6, '-k', label = ct_fitlabel)   
    plt.title("Qi vs T, " + str(power)+' dBm')    
    plt.xlabel('T (mK)')
    plt.ylabel('Internal Quality Factor Qi (1e6)')
    plt.legend()
    
    plt.show()
    return

def compare_plots_f0(results, results2, label1, label2, fr1, fr2, show_MB=True):
    plt.figure(1,figsize=(8,8))
    plt.plot(results[:,0],(results[:,1]/1e3),'ob',label = label1)
    plt.plot(results2[:,0],(results2[:,1]/1e3),'vr',label = label2)
    if show_MB:
        MB_f01=calc_MB_theory(results, fr1,'f0')
        MB_f02=calc_MB_theory(results2, fr2,'f0')
        plt.plot(results[:,0],MB_f01*1e-9,'-b',label=(label1+' fit'))
        plt.plot(results2[:,0],MB_f02*1e-9,'-r',label=(label2+' fit'))
    plt.title("f0 vs T of " + resonator + " resonator at " + str(power)+' dBm showing filter effect')
    plt.xlabel('T (mK)')
    plt.ylabel('f0 (GHz)')
    plt.legend()
    plt.show()
    return

def compare_plots_df0_f0(results, results2, label1, label2, fr1, fr2, show_MB=True):
    plt.figure(1,figsize=(8,8))
    plt.plot(results[:,0],(results[:,12]),'ob',label = label1)
    plt.plot(results2[:,0],(results2[:,12]),'vr',label = label2)
    if show_MB:
        MB_df_f01=calc_MB_theory(results, fr1,'df/f0')
        MB_df_f02=calc_MB_theory(results2, fr2,'df/f0')
        plt.plot(results[:,0],MB_df_f01,'-b',label=(label1+' fit'))
        plt.plot(results2[:,0],MB_df_f02,'-r',label=(label2+' fit'))
    plt.title("df0/f0 vs T of " + resonator + " resonator at " + str(power)+' dBm')
    plt.xlabel('T (mK)')
    plt.ylabel('df0/f0')
    plt.legend()
    plt.show()
    return

def calc_df0_f0(results,f0):
    df0_f0=(results[:,1]/1e3-f0)/f0
    #Convert into the right type of array to be appended line by line to results
    df0_f0=np.transpose(np.atleast_2d(df0_f0))
    #Append
    results=np.append(results,df0_f0,axis=1)
    return results

def calc_qi1(results):
    #Uses real part of Qc
    #WARNING:reciprocal will return 0 for any integer inputs greater than 1
    inv_real_qc=np.reciprocal(results[:,3])
    inv_qr=np.reciprocal(results[:,2])
    qis=np.reciprocal(inv_qr-inv_real_qc)
    #Convert into the right type of array to be appended line by line to results
    qis=np.transpose(np.atleast_2d(qis))
    #Append
    results=np.append(results,qis,axis=1)
    return results

def calc_qi2(results):
    #Uses magnitude of Qc, not yet formatted for pymkid data format
    mag_qc=np.sqrt(np.square(results[:,4])+np.square(results[:,9]))
    #WARNING:reciprocal will return 0 for any integer inputs greater than 1
    inv_real_qc=np.reciprocal(mag_qc)
    inv_qr=np.reciprocal(results[:,2])
    results[:,3]=np.reciprocal(inv_qr-inv_real_qc)
    return results

def calc_MB_theory(results, fr, select='both'):
    temps=results[:,0]
    #print(temps)
    MB_qis=[]
    MB_f0s=[]
    for i in range(len(temps)):
        MB_qis.append(MB_equations.Qi_T(temps[i]*1e-3, fr[0]*1e9,fr[3],fr[1]*1e-3,fr[2]))
        MB_f0s.append(MB_equations.f_T(temps[i]*1e-3, fr[0]*1e9,fr[1]*1e-3,fr[2]))
    #print(MB_qis)
    #print(MB_f0s)
    if select=='qi':
        return np.array(MB_qis)
    elif select=='f0':
        return np.array(MB_f0s)
    elif select=='df/f0':
        df_f0s=(np.array(MB_f0s)-fr[0]*1e9)/(fr[0]*1e9)
        #print(df_f0s)
        return df_f0s
    elif select=='both':
        return np.array(MB_qis),np.array(MB_f0s)

def pbp_Qi_benefit(results1, results2,chopval=0):
    #point by point calculation of Qi with filter/Qi without filter
    #results1 should be with filters, results2 without
    sum=0.
    for i in range(len(results1[:,11])-chopval):
        sum += results1[i,11]/results2[i,11]-1
    return (sum/(len(results1[:,11]-chopval)))

def plot_caltech_data(plotData=True):
    csv_fname='OW200127.mattis_bardeen_caltech.csv'
    with open(csv_fname, mode='r') as csvfile:
        data=[]
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            data.append(row)
        ktemps=data[0]
        temps=np.array([float(i)*1000 for i in data[0]])#in mK
        f0s=np.array([float(i) for i in data[2]])#in Hz
        qis=np.array([float(i) for i in data[4]])
    #do fits
    #frs stands for "fit results"
    frs = np.array(MB_equations.MB_fitter3(temps*1e-3,qis,f0s))
    
    df_f0s=((f0s*1e-9)-frs[0])/frs[0]
    MB_qis=[]
    MB_f0s=[]
    for i in range(len(temps)):
        MB_qis.append(MB_equations.Qi_T(temps[i]*1e-3, frs[0]*1e9,frs[3],frs[1]*1e-3,frs[2]))
        MB_f0s.append(MB_equations.f_T(temps[i]*1e-3, frs[0]*1e9,frs[1]*1e-3,frs[2]))
    MB_df_f0s=(np.array(MB_f0s)-frs[0]*1e9)/(frs[0]*1e9)


    #Get display results
    frs[2]=frs[2]*100 #convert alpha to percent
    dfr2=[]#display fit results
    for i in range(len(frs)):
        val = str(round(frs[i],(6-int(math.floor(math.log10(abs(frs[i])))) - 1)))#first int is how many sig figs you want displayed
        dfr2.append(val)
    print("Caltech data fit results are f0 = "+dfr2[0]+" Ghz, Delta0 = "+dfr2[1]+" meV, alpha is "+dfr2[2]+"%, Qi0 is "+dfr2[3]+", Chi-squared-dof is "+dfr2[4])
    frs[2]=frs[2]/100 #convert alpha back from percent
    if plotData:
        plt.figure(1,figsize=(12,6))
        matplotlib.rcParams['legend.loc']='lower left'

        plt.subplot(1,2,1)
        plt.plot(temps, df_f0s, 'ok', label = "Caltech Data")
        plt.plot(temps, MB_df_f0s, '-k', label = "Caltech MB Fit")
        plt.title("df0/f0 vs T")
        plt.xlabel('T (mK)')
        plt.ylabel('df0/f0')
        plt.legend()
        
        plt.subplot(1,2,2)
        fitlabel2='Caltech MB Fit:\n'+r'f$_0$ = '+ dfr2[0]+' Ghz\n'+r'Q$_0$ = '+dfr2[3]+'\n'+r'$\Delta$ = '+dfr2[1]+' meV\n'+r'$\alpha = $'+dfr2[2]+'%'
        plt.plot(temps,qis*1e-6,'ok',label='Caltech Data')
        plt.plot(temps,np.array(MB_qis)*1e-6,'-k',label=fitlabel2)
        
        plt.title("Qi vs T")
        plt.xlabel('T (mK)')
        plt.ylabel('Internal Quality Factor Qi (1e6)')
        plt.legend()
        
        plt.show()
    return np.array(temps), np.array(f0s), np.array(df_f0s), np.array(qis), np.array(MB_df_f0s), np.array(MB_qis), np.array(dfr2)

if __name__ == "__main__":
    datapath='/data/TempSweeps/'
    date='20210302'
    time='213525' #of the original sweep
    resonator = "Al"
    power = -70
    fit_datetime="20210430_1545"
    label = "With Filters"

    Round_Temps = True
    outlier_temps = [134]#this is for -50 dB
    Remove_Outliers = False
    Fit_Fits = True

    date2='20201222'
    time2='152824'
    resonator2 = "Al"
    fit_datetime2 = "20210430_1520"
    label2 = "Without Filters"
    outlier_temps2=[]

    results=read_data_PyMKID(datapath+date+"/PyMKID_fits/"+fit_datetime+"/P"+str(power)+"_"+fit_datetime+".txt")
    results2=read_data_PyMKID(datapath+date2+"/PyMKID_fits/"+fit_datetime2+"/P"+str(power)+"_"+fit_datetime2+".txt")
    #Calculate Qi from Qr and Qc (real part)
    results=calc_qi1(results)
    results2=calc_qi1(results2)
    #Clean up data, ex. by removing outliers and rounding temps to nearest 5 mK (function that records data truncates instead of rounds, so 94.989 mK --> 94.0 mK, needs fix)
    if Remove_Outliers:
        if len(outlier_temps) != 0:
            results = delete_outliers(results,outlier_temps)
        if len(outlier_temps2) != 0:
            results2 = delete_outliers(results2,outlier_temps2)

    if Round_Temps:
        results[:,0] = round_temps(results[:,0])
        results2[:,0] = round_temps(results2[:,0])
    
    c_temps, c_fs, c_dfs, c_qis, c_mb_dfs, c_mb_qis, dfr_ct = plot_caltech_data(False)

    if Fit_Fits:
        
        fr1,dfr1=display_MBfitresults(results,label)
        fr2,dfr2=display_MBfitresults(results2,label2)

        improvement = fr1[3]/fr2[3]*100-100
        #pbp_imp1=100*pbp_Qi_benefit(results,results2)
        #pbp_imp2=100*pbp_Qi_benefit(results,results2,15)
        #print("Filter improves fitted Qi0 by "+str(improvement)[0:4]+"%, point-by-point Qi by an average of "+str(pbp_imp1)[0:4]+"% generally and by "+str(pbp_imp2)[0:4]+"% at low temperatures.")
        print("Filter improves fitted Qi0 by "+str(improvement)[0:4]+"%")
    #Calculate (f0-f0_base)/f0_base, append to results
        results=calc_df0_f0(results,fr1[0])
        results2=calc_df0_f0(results2,fr2[0])
    #plot_df0_f0(results2,power,resonator2)
    #plot_qr(results,power,resonator)
    #compare_plots_qi(results,results2,label,label2, fr1, fr2)
    #compare_plots_f0(results,results2,label,label2,fr1,fr2)
    #compare_plots_df0_f0(results,results2,label,label2,fr1,fr2)
    
    masterplot(results,results2,label,label2,fr1,fr2,dfr1,dfr2,True,True,c_temps, c_dfs, c_qis, c_mb_dfs, c_mb_qis, dfr_ct)
