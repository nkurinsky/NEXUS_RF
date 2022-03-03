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

Nb6_NoSource = [
"20220225_072810",
"20220225_081557",
"20220225_090339",
"20220225_095120",
"20220225_103905"]

Nb7_NoSource = [
"20220225_131138",
"20220225_135931",
"20220225_144720"]

Nb6_WithSource = [
"20220227_190259",
"20220227_195051",
"20220227_203842"]

Nb7_WithSource = [
"20220227_150202",
"20220227_155000",
"20220227_163805",
"20220227_172553",
"20220227_181401"]

RunSets   = np.array([Nb6_NoSource,Nb7_NoSource,Nb6_WithSource,Nb7_WithSource], dtype=object)
RunLabels = np.array(["Nb6 No Source", "Nb7 No Source", "Nb6 With Source", "Nb7 With Source"])

datapath = "/data/ProcessedOutputs/"

## Create the figures
# fig1 = plt.figure(1)
# ax10 = plt.gca()
# ax10.set_xlabel('Applied RF Power [dBm]')
# ax10.set_ylabel(r'Resonator frequency $f$ [Hz]')

fig2 = plt.figure(2)
ax20 = plt.gca()
ax20.set_xlabel('Applied RF Power [dBm]')
ax20.set_ylabel(r'$(\langle f \rangle - f)/f$')

fig3 = plt.figure(3, figsize=(8,5))
ax30 = plt.gca()
ax30.set_xlabel('Applied RF Power [dBm]')
ax30.set_ylabel(r'Resonator Quality Factor $Q$')

fig4 = plt.figure(4)
ax40 = plt.gca()
ax40.set_xlabel('Applied RF Power [dBm]')
ax40.set_ylabel(r'$($max$(Q) - Q)/Q$')

## Do the same thing to every set of runs
for i in np.arange(len(RunSets)):

	## Output containers
	run_powers  = np.zeros(36)
	run_mean_Fs = np.zeros(36)
	run_mean_Qs = np.zeros(36)

	## Loop over every file in a single run set
	n_runs = len(RunSets[i])
	for j in np.arange(n_runs):

		fullpath = os.path.join(datapath,"out_"+RunSets[i][j])
		filename = "ResonanceFits_"+RunSets[i][j]+".h5"

		try:
			fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
		except:
			print("Problem with file:",os.path.join(fullpath,filename))
			print("Skipping...")
			continue

		## Pull out the result arrays
		run_powers   = fdata.powers
		run_mean_Fs += fdata.fit_fr
		run_mean_Qs += fdata.fit_Qr

	run_mean_Fs = run_mean_Fs/n_runs
	run_mean_Qs = run_mean_Qs/n_runs

	run_RMS_Fs  = np.sqrt(np.sum(np.power(run_mean_Fs-np.mean(run_mean_Fs),2)))
	run_RMS_Qs  = np.sqrt(np.sum(np.power(run_mean_Qs-np.mean(run_mean_Qs),2)))

	# ax10.plot(run_powers,run_mean_Fs, alpha=1.0, label=RunLabels[i])

	ax20.plot(run_powers,(np.mean(run_mean_Fs)-run_mean_Fs)/run_mean_Fs, alpha=1.0, label=RunLabels[i])

	ax30.plot(run_powers,run_mean_Qs, alpha=1.0, label=RunLabels[i])

	ax40.plot(run_powers,(np.max(run_mean_Qs)-run_mean_Qs)/run_mean_Qs, alpha=1.0, label=RunLabels[i])


# for i in np.arange(len(Nb6_NoSource)):

# 	fullpath = os.path.join(datapath,"out_"+Nb6_NoSource[i])
# 	filename = "ResonanceFits_"+Nb6_NoSource[i]+".h5"

# 	try:
# 		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
# 	except:
# 		print("Problem with file:",os.path.join(fullpath,filename))
# 		print("Skipping...")
# 		continue

# 	ax10.plot(fdata.powers,fdata.fit_fr, alpha=0.5)

# 	ax20.plot(fdata.powers,(np.mean(fdata.fit_fr)-fdata.fit_fr)/fdata.fit_fr, alpha=0.5)

# 	ax30.plot(fdata.powers,fdata.fit_Qr, alpha=0.5)

# for i in np.arange(len(Nb7_NoSource)):

# 	fullpath = os.path.join(datapath,"out_"+Nb7_NoSource[i])
# 	filename = "ResonanceFits_"+Nb7_NoSource[i]+".h5"

# 	try:
# 		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
# 	except:
# 		print("Problem with file:",os.path.join(fullpath,filename))
# 		print("Skipping...")
# 		continue

# 	ax10.plot(fdata.powers,fdata.fit_fr, alpha=0.5)

# 	ax20.plot(fdata.powers,(np.mean(fdata.fit_fr)-fdata.fit_fr)/fdata.fit_fr, alpha=0.5)

# 	ax30.plot(fdata.powers,fdata.fit_Qr, alpha=0.5)

# for i in np.arange(len(Nb6_WithSource)):

# 	fullpath = os.path.join(datapath,"out_"+Nb6_WithSource[i])
# 	filename = "ResonanceFits_"+Nb6_WithSource[i]+".h5"

# 	try:
# 		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
# 	except:
# 		print("Problem with file:",os.path.join(fullpath,filename))
# 		print("Skipping...")
# 		continue

# 	ax10.plot(fdata.powers,fdata.fit_fr, alpha=0.5)

# 	ax20.plot(fdata.powers,(np.mean(fdata.fit_fr)-fdata.fit_fr)/fdata.fit_fr, alpha=0.5)

# 	ax30.plot(fdata.powers,fdata.fit_Qr, alpha=0.5)

# for i in np.arange(len(Nb7_WithSource)):

# 	fullpath = os.path.join(datapath,"out_"+Nb7_WithSource[i])
# 	filename = "ResonanceFits_"+Nb7_WithSource[i]+".h5"

# 	try:
# 		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
# 	except:
# 		print("Problem with file:",os.path.join(fullpath,filename))
# 		print("Skipping...")
# 		continue

# 	ax10.plot(fdata.powers,fdata.fit_fr, alpha=0.5)

# 	ax20.plot(fdata.powers,(np.mean(fdata.fit_fr)-fdata.fit_fr)/fdata.fit_fr, alpha=0.5)

# 	ax30.plot(fdata.powers,fdata.fit_Qr, alpha=0.5)

# fig1.gca() ; plt.tight_layout() ; fig1.legend(loc='lower left') ; fig1.savefig("/home/nexus-admin/Downloads/Figure_1.png")
fig2.gca() ; plt.tight_layout() ; fig2.legend(loc='lower left') ; fig2.savefig("/home/nexus-admin/Downloads/Figure_2.png")
fig3.gca() ; plt.tight_layout() ; fig3.legend(loc='lower left') ; fig3.savefig("/home/nexus-admin/Downloads/Figure_3.png")
fig4.gca() ; plt.tight_layout() ; fig4.legend(loc='lower left') ; fig4.savefig("/home/nexus-admin/Downloads/Figure_4.png")
plt.show()

