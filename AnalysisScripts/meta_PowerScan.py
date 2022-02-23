import os, h5py
import numpy as np
import matplotlib.pyplot as plt
import ResonanceFitResult as fit

## Set up matplotlib options for plots
plt.rcParams['axes.grid'] = True
plt.rcParams.update({'font.size': 12})
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
dfc = plt.rcParams['axes.prop_cycle'].by_key()['color']

date = "20220214"
sers = "220330"

series = date + "_" + sers

datapath = "/data/ProcessedOutputs/VNA/"

fullpath = os.path.join(datapath,"out_"+series)
filename = "ResonanceFits_"+series+".h5"

fdata = fit.decode_hdf5(os.path.join(fullpath,filename))

fig = plt.figure()
plt.plot(power_list,fr_list)
plt.xlabel('Applied RF Power [dBm]')
plt.ylabel(r'Resonator frequency $f$ [Hz]')
fig.savefig(os.path.join(out_path,"f_vs_P.png"), format='png')

fig = plt.figure()
plt.plot(power_list,(np.mean(fr_list)-fr_list)/fr_list)
plt.xlabel('Applied RF Power [dBm]')
plt.ylabel(r'$\Delta f/f$')
fig.savefig(os.path.join(out_path,"df_vs_P.png"), format='png')

fig = plt.figure()
plt.plot(power_list,Qr_list)
plt.xlabel('Applied RF Power [dBm]')
plt.ylabel(r'Resonator Quality Factor $Q$')
fig.savefig(os.path.join(out_path,"Q_vs_P.png"), format='png')

plt.show()

