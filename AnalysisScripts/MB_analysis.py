from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import ResonanceFitter as fitres
import MB_equations as MBe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

def fit_temp_res(fname, sweepnum, pickedres=None):
    with h5py.File(fname, 'r') as fyle:
        timestamps = fyle['tempsweep'].keys()
        chosen_tempsweep = timestamps[sweepnum] # Just the first temperature sweep saved in this file
        print(chosen_tempsweep)
        MCTemps1 = np.array(fyle['tempsweep/'+chosen_tempsweep].keys())
        MCTemps1 = MCTemps1[MCTemps1!='MB']
        MCTemps1 = MCTemps1[MCTemps1!='RES']

    df1 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/'+MCTemps1[0])

    if pickedres is not None:
        if isinstance(pickedres, int):
            MKIDnum = [pickedres]
        else:
            MKIDnum = pickedres
    else:
        MKIDnum = np.unique(df1['resID'])

    ID_of_T = np.zeros((len(MCTemps1),len(MKIDnum)))
    fr_of_T = np.zeros((len(MCTemps1),len(MKIDnum)))
    Qr_of_T = np.zeros((len(MCTemps1),len(MKIDnum)))
    Qc_of_T = np.zeros((len(MCTemps1),len(MKIDnum)))
    a_of_T = (1+1j)*np.zeros((len(MCTemps1),len(MKIDnum)))
    phi_of_T = np.zeros((len(MCTemps1),len(MKIDnum)))
    tau_of_T = (1+1j)*np.zeros((len(MCTemps1),len(MKIDnum)))
    Qc_hat_mag_of_T = np.zeros((len(MCTemps1),len(MKIDnum)))

    print('Doing resonance fits...')

    for Tn in range(len(MCTemps1)):
        temperature_Tn = MCTemps1[Tn]
        df1 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)
        resID_index = df1['resID']

        for Kn in range(len(MKIDnum)):
            fr_0 = np.mean(df1['f0'][resID_index==MKIDnum[Kn]][0])
            if Tn>0:
                if fr_of_T[Tn-1,Kn]!=0:
                    fr_0 = fr_of_T[Tn-1,Kn]
            f1 = df1['f']
            f_array = np.array(f1[resID_index==MKIDnum[Kn]])
            z1 = df1['z']
            z_array = np.array(z1[resID_index==MKIDnum[Kn]])

            try:
                fitted = fitres.finefit(f_array, z_array, fr_0)
            except:
                plt.plot(f_array,abs(z_array))
                plt.title('MKID '+str(MKIDnum[Kn])+', '+str(MCTemps1[Tn])+' K')
                plt.show()
                fitted = [0,0,0,0,0,0,0]

            ID_of_T[Tn,Kn] = MKIDnum[Kn]
            fr_of_T[Tn,Kn] = fitted[0]
            Qr_of_T[Tn,Kn] = fitted[1]
            Qc_of_T[Tn,Kn] = fitted[6]
            a_of_T[Tn,Kn] = fitted[3]
            phi_of_T[Tn,Kn] = fitted[4]
            tau_of_T[Tn,Kn] = fitted[5]
            Qc_hat_mag_of_T[Tn,Kn] = fitted[2]

    df2 = pd.DataFrame()
    df2['resID'] = ID_of_T.flatten(order='F')
    df2['fr'] = fr_of_T.flatten(order='F')
    df2['Qr'] = Qr_of_T.flatten(order='F')
    df2['Qc'] = Qc_of_T.flatten(order='F')
    df2['a'] = a_of_T.flatten(order='F')
    df2['phi'] = phi_of_T.flatten(order='F')
    df2['tau'] = tau_of_T.flatten(order='F')
    df2['Qc_hat_mag'] = Qc_hat_mag_of_T.flatten(order='F')
    df2.to_hdf(fname, key='/tempsweep/'+chosen_tempsweep+'/RES')

def plot_res(fname, sweepnum, pickedres=None, title1='', show=True):
    with h5py.File(fname, 'r') as fyle:
        timestamps = fyle['tempsweep'].keys()
        chosen_tempsweep = timestamps[sweepnum]
        print(chosen_tempsweep)
        MCTemps1 = np.array(fyle['tempsweep/'+chosen_tempsweep].keys())
        MCTemps1 = MCTemps1[MCTemps1!='MB']
        MCTemps1 = MCTemps1[MCTemps1!='RES']

    df2 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/RES')

    if pickedres is not None:
        if isinstance(pickedres, int):
            MKIDnum = [pickedres]
        else:
            MKIDnum = pickedres
    else:
        MKIDnum = np.unique(df2['resID'])

    cmap = plt.cm.jet
    norm = Normalize(vmin=0, vmax=max(np.array(MCTemps1,dtype=float)))

    print('Plotting resonance fits...')

    for Kn in range(len(MKIDnum)):
        param_array = np.array(df2.loc[df2['resID'] == MKIDnum[Kn]])

        for Tn in range(len(MCTemps1)):
            params = param_array[Tn][1:]
            temperature_Tn = MCTemps1[Tn]
            df1 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)

            if params[0]==0:
                print('failed fit: '+'tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)
                continue
            if params[2] < 0:
                print('negative Qc: '+'tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)
            if params[1]*params[2]/(params[2]-params[1]) < 0:
                print('negative Qi: '+'tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)

            resID_index = df1['resID']
            f1 = df1['f']
            f_array = np.array(f1[resID_index==MKIDnum[Kn]])
            z1 = df1['z']
            z_array = np.array(z1[resID_index==MKIDnum[Kn]])
            f_array_fill = np.linspace(f_array[0],f_array[-1],10*len(f_array))
            fit_z_uncorrected = fitres.resfunc3(f_array_fill, params[0], params[1], params[6], params[3], params[4], params[5])
            fit_z_corrected = 1-(params[1]/params[2])/(1+2j*params[1]*(f_array_fill-params[0])/params[0])
            z_array_corrected = 1-((1-z_array/(params[3]*np.exp(-2j*np.pi*(f_array-params[0])*params[5])))*(np.cos(params[4])/np.exp(1j*params[4])))

            if Tn==0:
                temp_label=temperature_Tn+' K'+title1
                fit_label=None
            elif Tn==len(MCTemps1)-1:
                temp_label=temperature_Tn+' K'+title1
                fit_label='fit'
            else:
                temp_label=None
                fit_label=None

            plt.figure(1)
            plt.plot(f_array, 20*np.log10(np.abs(z_array)), c=cmap(norm(temperature_Tn.astype(np.float))), label=temp_label, zorder=0-Tn)
            plt.plot(f_array_fill, 20*np.log10(np.abs(fit_z_uncorrected)), c='k', zorder=1, linewidth=1, linestyle='--', label=fit_label)

            plt.figure(2)
            plt.plot(z_array.real, z_array.imag, c=cmap(norm(temperature_Tn.astype(np.float))), label=temp_label, zorder=0-Tn)
            plt.plot(fit_z_uncorrected.real, fit_z_uncorrected.imag, c='k', zorder=1, linewidth=1, linestyle='--', label=fit_label)

            plt.figure(3)
            plt.plot(z_array_corrected.real, z_array_corrected.imag, c=cmap(norm(temperature_Tn.astype(np.float))), label=temp_label, zorder=0-Tn)
            plt.plot(fit_z_corrected.real, fit_z_corrected.imag, c='k', zorder=1, linewidth=1, linestyle='--', label=fit_label)

        plt.figure(1)
        plt.title('Resonance #'+str(int(MKIDnum[Kn])))
        plt.xlabel('frequency [GHz]')
        plt.ylabel('absolute transmission [dB]')
        plt.legend()

        plt.figure(2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axvline(x=0, c='gray', zorder=-1*len(MCTemps1))
        plt.axhline(y=0, c='gray', zorder=-1*len(MCTemps1))
        plt.title('Resonance #'+str(int(MKIDnum[Kn])))
        plt.xlabel('real')
        plt.ylabel('imag')
        plt.legend()

        plt.figure(3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axvline(x=0, c='gray', zorder=-1*len(MCTemps1))
        plt.axhline(y=0, c='gray', zorder=-1*len(MCTemps1))
        plt.title('Resonance #'+str(int(MKIDnum[Kn])))
        plt.xlabel('real')
        plt.ylabel('imag')
        plt.legend()

        if show:
            plt.show()

def fit_temp_MB(fname, sweepnum, pickedres=None):
    with h5py.File(fname, 'r') as fyle:
        timestamps = fyle['tempsweep'].keys()
        chosen_tempsweep = timestamps[sweepnum]
        print(chosen_tempsweep)
        MCTemps1 = np.array(fyle['tempsweep/'+chosen_tempsweep].keys())
        MCTemps1 = MCTemps1[MCTemps1!='MB']
        MCTemps1 = MCTemps1[MCTemps1!='RES']

    df2 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/RES')

    if pickedres is not None:
        if isinstance(pickedres, int):
            MKIDnum = [pickedres]
        else:
            MKIDnum = pickedres
    else:
        MKIDnum = np.unique(df2['resID'])

    f0 = np.zeros(len(MKIDnum))
    Delta = np.zeros(len(MKIDnum))
    alpha = np.zeros(len(MKIDnum))
    Qi0 = np.zeros(len(MKIDnum))

    MCTemps1 = np.array(MCTemps1, dtype=float)

    print('Doing Mattis-Bardeen fits...')
    for Kn in range(len(MKIDnum)):
        print(Kn)
        fr_of_T = np.array(df2['fr'][df2['resID']==Kn])
        Qr_of_T = np.array(df2['Qr'][df2['resID']==Kn])
        Qc_of_T = np.array(df2['Qc'][df2['resID']==Kn])

        T_fitter = MCTemps1[fr_of_T!=0]
        fr_fitter = fr_of_T[fr_of_T!=0]*1e9
        Qi_fitter = Qr_of_T[fr_of_T!=0]*Qc_of_T[fr_of_T!=0]/(Qc_of_T[fr_of_T!=0]-Qr_of_T[fr_of_T!=0])

        results = MBe.MB_fitter(T_fitter,Qi_fitter,fr_fitter)

        f0[Kn] = results[0]
        Delta[Kn] = results[1]
        alpha[Kn] = results[2]
        Qi0[Kn] = results[3]

    df3 = pd.DataFrame()
    df3['resID'] = MKIDnum
    df3['f0'] = f0
    df3['Delta'] = Delta
    df3['alpha'] = alpha
    df3['Qi0'] = Qi0
    df3.to_hdf(fname, key='/tempsweep/'+chosen_tempsweep+'/MB')

def plot_MB(fname, sweepnum, pickedres=None, show=True):
    with h5py.File(fname, 'r') as fyle:
        timestamps = fyle['tempsweep'].keys()
        chosen_tempsweep = timestamps[sweepnum]
        print(chosen_tempsweep)
        MCTemps1 = np.array(fyle['tempsweep/'+chosen_tempsweep].keys())
        MCTemps1 = MCTemps1[MCTemps1!='MB']
        MCTemps1 = MCTemps1[MCTemps1!='RES']

    df2 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/RES')
    resID_index = df2['resID']
    if pickedres is not None:
        if isinstance(pickedres, int):
            MKIDnum = [pickedres]
        else:
            MKIDnum = pickedres
    else:
        MKIDnum = np.unique(np.array(resID_index,dtype=int))

    df3 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/MB')

    MCTemps1 = np.array(MCTemps1, dtype=float)
    T_smooth = np.linspace(0.0001,MCTemps1[-1],10000)

    print('Plotting Mattis-Bardeen fits...')
    for Kn in range(len(MKIDnum)):
        fr_of_T = np.array(df2['fr'][df2['resID']==MKIDnum[Kn]])
        Qr_of_T = np.array(df2['Qr'][df2['resID']==MKIDnum[Kn]])
        Qc_of_T = np.array(df2['Qc'][df2['resID']==MKIDnum[Kn]])

        T_fitter = MCTemps1[fr_of_T!=0]
        fr_fitter = fr_of_T[fr_of_T!=0]*1e9
        Qi_fitter = Qr_of_T[fr_of_T!=0]*Qc_of_T[fr_of_T!=0]/(Qc_of_T[fr_of_T!=0]-Qr_of_T[fr_of_T!=0])

        f0 = np.array(df3['f0'])[MKIDnum[Kn]]*1e9
        Qi0 = np.array(df3['Qi0'])[MKIDnum[Kn]]
        Delta = np.array(df3['Delta'])[MKIDnum[Kn]]*1e-3
        alpha = np.array(df3['alpha'])[MKIDnum[Kn]]

        fig, axarr = plt.subplots(ncols=2, nrows=1, figsize=(12,5))

        axarr[0].semilogy(T_fitter*1000, 1e-6*fr_fitter/max(fr_fitter), 'o', markersize=8)#, label='f data')
        axarr[0].plot(T_smooth*1000, 1e-6*MBe.f_T(T_smooth,f0,Delta,alpha)/max(fr_fitter), '--', c='k', linewidth=3)#, label='f(T) fit' )
        axarr[0].set_title('Mattis Bardeen fit')
        axarr[0].set_xlabel('Temperature [mK]')
        axarr[0].set_ylabel('fr [MHz]')

        axarr[1].semilogy(T_fitter[Qi_fitter>0]*1000, Qi_fitter[Qi_fitter>0], 'o', markersize=8, label='Data')
        axarr[1].semilogy(T_fitter[Qi_fitter<0]*1000, -Qi_fitter[Qi_fitter<0], 'o', markersize=8, label='negative Qi Data')
        axarr[1].semilogy(T_smooth*1000, MBe.Qi_T(T_smooth,f0,Qi0,Delta,alpha), '--', c='k', label='Fit:', linewidth=3)
        axarr[1].set_title('Resonance #'+str(int(MKIDnum[Kn])))
        axarr[1].set_xlabel('Temperature [mK]')
        axarr[1].set_ylabel('Qi')
        axarr[1].plot([], [], ' ', label='f$_{0}$ = %.3f GHz\nQ$_{i0}$ = %.3f\n$\Delta$ = %.3f meV\n'%(f0/1.e9,Qi0,Delta*1000.)+r'$\alpha$'+' = %.3f'%(100*alpha)+'%')

        plt.legend()

        if show:
            plt.show()

def MB_kappa(fname, sweepnum, pickedres=None):
    with h5py.File(fname, 'r') as fyle:
        timestamps = fyle['tempsweep'].keys()
        chosen_tempsweep = timestamps[sweepnum]
        print(chosen_tempsweep)

    df2 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/RES')
    resID_index = df2['resID']
    if pickedres is not None:
        if isinstance(pickedres, int):
            MKIDnum = [pickedres]
        else:
            MKIDnum = pickedres
    else:
        MKIDnum = np.unique(np.array(resID_index,dtype=int))

    df3 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/MB')

    return np.array([df3['f0'],df3['Delta'],df3['alpha'],df3['Qi0']]).T[MKIDnum]

def save_MB(fname, sweepnum, pickedres=None, show=False):
    with h5py.File(fname, 'r') as fyle:
        timestamps = fyle['tempsweep'].keys()
        chosen_tempsweep = timestamps[sweepnum]
        print(chosen_tempsweep)
        MCTemps1 = np.array(fyle['tempsweep/'+chosen_tempsweep].keys())
        MCTemps1 = MCTemps1[MCTemps1!='MB']
        MCTemps1 = MCTemps1[MCTemps1!='RES']

    df2 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/RES')
    resID_index = df2['resID']
    if pickedres is not None:
        if isinstance(pickedres, int):
            MKIDnum = [pickedres]
        else:
            MKIDnum = pickedres
    else:
        MKIDnum = np.unique(np.array(resID_index,dtype=int))

    df3 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/MB')

    MCTemps_f = np.array(MCTemps1, dtype=float)
    T_smooth = np.linspace(0.0001,MCTemps_f[-1],10000)

    cmap = plt.cm.jet
    norm = Normalize(vmin=0, vmax=max(np.array(MCTemps1,dtype=float)))

    Res_pdf = PdfPages(fname[:-3]+'_MB.pdf')

    print('Plotting Mattis-Bardeen fits...')
    for Kn in range(len(MKIDnum)):
        param_array = np.array(df2.loc[df2['resID'] == MKIDnum[Kn]])
        fig, axarr = plt.subplots(ncols=2, nrows=2, figsize=(10,10))

        for Tn in range(len(MCTemps1)):
            params = param_array[Tn][1:]
            temperature_Tn = MCTemps1[Tn]
            df1 = pd.read_hdf(fname, key='tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)

            if params[0]==0:
                print('failed fit: '+'tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)
                continue
            if params[2] < 0:
                print('negative Qc: '+'tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)
            if params[1]*params[2]/(params[2]-params[1]) < 0:
                print('negative Qi: '+'tempsweep/'+chosen_tempsweep+'/'+temperature_Tn)

            resID_index = df1['resID']
            f1 = df1['f']
            f_array = np.array(f1[resID_index==MKIDnum[Kn]])
            z1 = df1['z']
            z_array = np.array(z1[resID_index==MKIDnum[Kn]])
            f_array_fill = np.linspace(f_array[0],f_array[-1],10*len(f_array))
            fit_z_uncorrected = fitres.resfunc3(f_array_fill, params[0], params[1], params[6], params[3], params[4], params[5])
            fit_z_corrected = 1-(params[1]/params[2])/(1+2j*params[1]*(f_array_fill-params[0])/params[0])
            z_array_corrected = 1-((1-z_array/(params[3]*np.exp(-2j*np.pi*(f_array-params[0])*params[5])))*(np.cos(params[4])/np.exp(1j*params[4])))

            if Tn==0:
                temp_label=temperature_Tn+' K'
                fit_label=None
            elif Tn==len(MCTemps1)-1:
                temp_label=temperature_Tn+' K'
                fit_label='fit'
            else:
                temp_label=None
                fit_label=None

            axarr[1,0].plot(f_array, 20*np.log10(np.abs(z_array)), c=cmap(norm(temperature_Tn.astype(np.float))), label=temp_label, zorder=0-Tn)
            axarr[1,0].plot(f_array_fill, 20*np.log10(np.abs(fit_z_uncorrected)), c='k', zorder=1, linewidth=1, linestyle='--', label=fit_label)

            #plt.figure(2)
            #plt.plot(z_array.real, z_array.imag, c=cmap(norm(temperature_Tn.astype(np.float))), label=temp_label, zorder=0-Tn)
            #plt.plot(fit_z_uncorrected.real, fit_z_uncorrected.imag, c='k', zorder=1, linewidth=1, linestyle='--', label=fit_label)

            axarr[1,1].plot(z_array_corrected.real, z_array_corrected.imag, c=cmap(norm(temperature_Tn.astype(np.float))), label=temp_label, zorder=0-Tn)
            axarr[1,1].plot(fit_z_corrected.real, fit_z_corrected.imag, c='k', zorder=1, linewidth=1, linestyle='--', label=fit_label)

        #axarr[1,0].set_title('Resonance #'+str(int(MKIDnum[Kn])))
        axarr[1,0].set_xlabel('frequency [GHz]')
        axarr[1,0].set_ylabel('absolute transmission [dB]')
        axarr[1,0].legend()


        axarr[1,1].set_aspect('equal', adjustable='box')
        axarr[1,1].axvline(x=0, c='gray', zorder=-1*len(MCTemps1))
        axarr[1,1].axhline(y=0, c='gray', zorder=-1*len(MCTemps1))
        #axarr[1,1].set_title('Resonance #'+str(int(MKIDnum[Kn])))
        axarr[1,1].set_xlabel('real')
        axarr[1,1].set_ylabel('imag')
        axarr[1,1].legend()

        fr_of_T = np.array(df2['fr'][df2['resID']==MKIDnum[Kn]])
        Qr_of_T = np.array(df2['Qr'][df2['resID']==MKIDnum[Kn]])
        Qc_of_T = np.array(df2['Qc'][df2['resID']==MKIDnum[Kn]])

        T_fitter = MCTemps_f[fr_of_T!=0]
        fr_fitter = fr_of_T[fr_of_T!=0]*1e9
        Qi_fitter = Qr_of_T[fr_of_T!=0]*Qc_of_T[fr_of_T!=0]/(Qc_of_T[fr_of_T!=0]-Qr_of_T[fr_of_T!=0])

        f0 = np.array(df3['f0'])[MKIDnum[Kn]]*1e9
        Qi0 = np.array(df3['Qi0'])[MKIDnum[Kn]]
        Delta = np.array(df3['Delta'])[MKIDnum[Kn]]*1e-3
        alpha = np.array(df3['alpha'])[MKIDnum[Kn]]

        axarr[0,0].plot(T_fitter*1000, 1e-6*abs(fr_fitter), 'o', markersize=8)#, label='f data')
        axarr[0,0].plot(T_smooth*1000, 1e-6*abs(MBe.f_T(T_smooth,f0,Delta,alpha)), '--', c='k', linewidth=3)#, label='f(T) fit' )
        axarr[0,0].set_title('Mattis Bardeen fit')
        axarr[0,0].set_xlabel('Temperature [mK]')
        axarr[0,0].set_ylabel('fr [MHz]')

        axarr[0,1].semilogy(T_fitter[Qi_fitter>0]*1000, Qi_fitter[Qi_fitter>0], 'o', markersize=8, label='Data')
        axarr[0,1].semilogy(T_fitter[Qi_fitter<0]*1000, -Qi_fitter[Qi_fitter<0], 'o', markersize=8, label='negative Qi Data')
        axarr[0,1].semilogy(T_smooth*1000, MBe.Qi_T(T_smooth,f0,Qi0,Delta,alpha), '--', c='k', label='Fit:', linewidth=3)
        axarr[0,1].set_title('Resonance #'+str(int(MKIDnum[Kn])))
        axarr[0,1].set_xlabel('Temperature [mK]')
        axarr[0,1].set_ylabel('Qi')
        axarr[0,1].plot([], [], ' ', label='f$_{0}$ = %.3f GHz\nQ$_{i0}$ = %.3f\n$\Delta$ = %.3f meV\n'%(f0/1.e9,Qi0,Delta*1000.)+r'$\alpha$'+' = %.3f'%(100*alpha)+'%')
        axarr[0,1].legend()

        Res_pdf.savefig()
        if show:
            plt.show()
        else:
            plt.close()

    Res_pdf.close()

if __name__ == '__main__':

    #fit_temp_res('200114OW190920p2.h5',1,pickedres=None)
    #plot_res('200114OW190920p2.h5',1,pickedres=None)
    #fit_temp_MB('200114OW190920p2.h5',1,pickedres=None)
    plot_MB('200114OW190920p2.h5',1,pickedres=12)
    #save_MB('200114OW190920p2.h5',1,pickedres=None,show=False)
