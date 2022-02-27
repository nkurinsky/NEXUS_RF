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

# series_list = [ ## Al resonator, Sources in CR
# "20220214_220330",
# "20220214_225344",
# "20220214_234130",
# "20220215_002929",
# "20220215_011731",
# "20220215_020520",
# # "20220215_025315",
# # "20220215_034114",
# "20220215_042905",
# "20220215_051659"]

# series_list = [ ## Al resonator, No sources in CR
# "20220216_091922",
# "20220216_100716",
# "20220216_105505",
# "20220216_114303",
# "20220216_123052"]

# series_list = [ ## Al resonator, No sources in CR, Shield Closed
# "20220223_154540",
# "20220223_163334",
# "20220223_172123",
# "20220223_180912",
# "20220223_185707",
# "20220223_194457",
# "20220223_203246",
# "20220223_212036",
# "20220223_220831",
# "20220223_225623"]

series_list = [ ## Al resonator, Ba133+Cs137 sources in CR, Shield Closed
"20220225_193444",
"20220225_202236",
"20220225_211027",
"20220225_215820",
"20220225_224619",
"20220225_233412",
"20220226_002202",
"20220226_010945",
"20220226_015734",
"20220226_024521"]

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

for i in np.arange(len(series_list)):

	fullpath = os.path.join(datapath,"out_"+series_list[i])
	filename = "ResonanceFits_"+series_list[i]+".h5"

	try:
		fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
	except:
		print("Problem with file:",os.path.join(fullpath,filename))
		print("Skipping...")
		continue

	ax10.plot(fdata.powers,fdata.fit_fr, alpha=0.5)

	ax20.plot(fdata.powers,(np.mean(fdata.fit_fr)-fdata.fit_fr)/fdata.fit_fr, alpha=0.5)

	ax30.plot(fdata.powers,fdata.fit_Qr, alpha=0.5)

fig1.gca() ; plt.tight_layout() ; fig1.savefig("/home/nexus-admin/Downloads/Figure_1.png")
fig2.gca() ; plt.tight_layout() ; fig2.savefig("/home/nexus-admin/Downloads/Figure_2.png")
fig3.gca() ; plt.tight_layout() ; fig3.savefig("/home/nexus-admin/Downloads/Figure_3.png")
plt.show()

