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

datapath = "/data/ProcessedOutputs/"

## Create the figures
fig1 = plt.figure(1)
ax10 = plt.gca()
ax10.set_xlabel('Applied RF Power [dBm]')
ax10.set_ylabel(r'Resonator frequency $f$ [Hz]')

fig2 = plt.figure(2)
ax20 = plt.gca()
ax20.set_xlabel('Applied RF Power [dBm]')
ax20.set_ylabel(r'$(\langle f \rangle - f)/f$')

fig3 = plt.figure(3, figsize=(8,5))
ax30 = plt.gca()
ax30.set_xlabel('Applied RF Power [dBm]')
ax30.set_ylabel(r'Resonator Quality Factor $Q$')

for i in np.arange(len(Nb6_NoSource)):

	fullpath = os.path.join(datapath,"out_"+Nb6_NoSource[i])
	filename = "ResonanceFits_"+Nb6_NoSource[i]+".h5"

	try:
		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
	except:
		print("Problem with file:",os.path.join(fullpath,filename))
		print("Skipping...")
		continue

	ax10.plot(fdata.powers,fdata.fit_fr, alpha=0.5)

	ax20.plot(fdata.powers,(np.mean(fdata.fit_fr)-fdata.fit_fr)/fdata.fit_fr, alpha=0.5)

	ax30.plot(fdata.powers,fdata.fit_Qr, alpha=0.5)

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

for i in np.arange(len(Nb6_WithSource)):

	fullpath = os.path.join(datapath,"out_"+Nb6_WithSource[i])
	filename = "ResonanceFits_"+Nb6_WithSource[i]+".h5"

	try:
		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
	except:
		print("Problem with file:",os.path.join(fullpath,filename))
		print("Skipping...")
		continue

	ax10.plot(fdata.powers,fdata.fit_fr, alpha=0.5)

	ax20.plot(fdata.powers,(np.mean(fdata.fit_fr)-fdata.fit_fr)/fdata.fit_fr, alpha=0.5)

	ax30.plot(fdata.powers,fdata.fit_Qr, alpha=0.5)

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

fig1.gca() ; plt.tight_layout() ; fig1.savefig("/home/nexus-admin/Downloads/Figure_1.png")
fig2.gca() ; plt.tight_layout() ; fig2.savefig("/home/nexus-admin/Downloads/Figure_2.png")
fig3.gca() ; plt.tight_layout() ; fig3.savefig("/home/nexus-admin/Downloads/Figure_3.png")
plt.show()

