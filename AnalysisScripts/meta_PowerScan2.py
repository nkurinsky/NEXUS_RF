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

series_list = [
# "20220417_195127",
# "20220417_201631",
# "20220417_202641",
# "20220417_203605",
# "20220417_204109",
# "20220417_204429",
# "20220417_204734",
# "20220417_205034",
# "20220417_205655",
"20220417_210100",
"20220417_210238",
"20220417_210421",
"20220417_210600",
"20220417_210739",
"20220417_210921",
"20220417_211057",
"20220417_211234",
"20220417_211419",
"20220417_211558",
# "20220417_211810",
# "20220417_211950",
# "20220417_212126",
# "20220417_212309",
# "20220417_212450",
# "20220417_212627",
# "20220417_212804",
# "20220417_212943",
# "20220417_213119",
# "20220417_213302",
# "20220417_213946",
# "20220417_214318"
]

datapath = "/data/ProcessedOutputs/"

## Create the figures
fig1 = plt.figure(1)
ax10 = plt.gca()
ax10.set_xlabel('Applied RF Power [dBm]')
ax10.set_ylabel(r'Resonator frequency $f$ [Hz]')

fig2 = plt.figure(2)
ax20 = plt.gca()
ax20.set_xlabel('Applied RF Power [dBm]')
ax20.set_ylabel(r'$(f - \langle f \rangle)/f$')

fig3 = plt.figure(3, figsize=(8,5))
ax30 = plt.gca()
ax30.set_xlabel('Applied RF Power [dBm]')
ax30.set_ylabel(r'Resonator Quality Factor $Q$')

## Count the series
n_sers = len(series_list)

## Check how many powers are in the first series
powers = np.arange(start=-50,stop=-20+2,step=2)
n_pwrs = len(powers)


## Create some output containers
f_by_power = np.zeros(shape=(n_pwrs,n_sers))

n_series_loaded = 0

## Loop over each series in this class
for i in np.arange(len(series_list)):

	fullpath = os.path.join(datapath,"out_"+series_list[i])
	filename = "ResonanceFits_"+series_list[i]+".h5"

	try:
		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
	except:
		print("Problem with file:",os.path.join(fullpath,filename))
		print("Skipping...")
		f_by_power[i,:] = np.nan * np.ones(n_pwrs)
		Q_by_power[i,:] = np.nan * np.ones(n_pwrs)
		continue

	f_by_power[i,:] = fdata.fit_fr
	Q_by_power[i,:] = fdata.fit_Qr

	ax10.plot(fdata.powers, fdata.fit_fr, 
		alpha=0.5, color=colors[j], label=class_names[j] if i==0 else None)

	ax20.plot(fdata.powers, (fdata.fit_fr-np.mean(fdata.fit_fr))/fdata.fit_fr, 
		alpha=0.5, color=colors[j], label=class_names[j] if i==0 else None)

	ax30.plot(fdata.powers, fdata.fit_Qr,
		alpha=0.5, color=colors[j], label=class_names[j] if i==0 else None)

f_mean_by_power = np.mean(f_by_power, axis=1)
Q_mean_by_power = np.mean(Q_by_power, axis=1)

f_sdev_by_power = np.std( f_by_power, axis=1)
Q_sdev_by_power = np.std( Q_by_power, axis=1)

fig1.gca() ; fig1.errorbar(powers, f_mean_by_power, yerr=f_sdev_by_power, marker='o')
fig3.gca() ; fig3.errorbar(powers, Q_mean_by_power, yerr=Q_sdev_by_power, marker='o')

fig1.gca() ; ax10.legend(loc="best") ; plt.tight_layout() ; fig1.savefig("/home/nexus-admin/Downloads/Figure_1.png")
fig2.gca() ; ax20.legend(loc="best") ; plt.tight_layout() ; fig2.savefig("/home/nexus-admin/Downloads/Figure_2.png")
fig3.gca() ; ax30.legend(loc="best") ; plt.tight_layout() ; fig3.savefig("/home/nexus-admin/Downloads/Figure_3.png")
plt.show()

