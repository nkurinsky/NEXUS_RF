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

series_list = [ ## Al resonator, All
"20220203_135029",
"20220203_144253",
"20220203_145146",
"20220203_145315",
"20220203_150210",
"20220203_150507",
"20220203_151401",
"20220203_152254",
"20220203_152257",
"20220203_152300",
"20220203_153153",
"20220203_161023",
"20220203_173322",
"20220203_175016",
"20220203_180038",
"20220214_220330",
"20220214_225344",
"20220214_234130",
"20220215_002929",
"20220215_011731",
"20220215_020520",
"20220215_025315",
"20220215_034114",
"20220215_042905",
"20220215_051659",
"20220215_060501",
"20220216_091922",
"20220216_100716",
"20220216_105505",
"20220216_114303",
"20220216_123052",
"20220223_154540",
"20220223_163334",
"20220223_172123",
"20220223_180912",
"20220223_185707",
"20220223_194457",
"20220223_203246",
"20220223_212036",
"20220223_220831",
"20220223_225623",
"20220225_192309",
"20220225_193444",
"20220225_202236",
"20220225_211027",
"20220225_215820",
"20220225_224619",
"20220225_233412",
"20220226_002202",
"20220226_010945",
"20220226_015734",
"20220226_024521",
"20220227_110202",
"20220227_114953",
"20220227_123741",
"20220227_132544",
"20220227_141329",
"20220302_174251",
"20220302_183051",
"20220302_191845",
"20220302_200631",
"20220302_205426",
"20220304_094854",
"20220317_094420",
"20220318_001948"]

series_list_1 = [ ## Al resonator, Ba133+Cs137 sources in CR
"20220214_220330",
"20220214_225344",
"20220214_234130",
"20220215_002929",
"20220215_011731",
"20220215_020520",
"20220215_025315",
"20220215_034114",
"20220215_042905",
"20220215_051659",
"20220215_060501",
"20220216_091922",
"20220216_100716",
"20220216_105505",
"20220216_114303",
"20220216_123052",
"20220223_154540",
"20220223_163334",
"20220223_172123",
"20220223_180912",
"20220223_185707",
"20220223_194457",
"20220223_203246",
"20220223_212036",
"20220223_220831",
"20220223_225623",
"20220225_192309",
"20220225_193444",
"20220225_202236",
"20220225_211027",
"20220225_215820",
"20220225_224619",
"20220225_233412",
"20220226_002202",
"20220226_010945",
"20220226_015734",
"20220226_024521",
"20220227_110202",
"20220227_114953",
"20220227_123741",
"20220227_132544",
"20220227_141329"]

series_list_2 = [ ## Al resonator, No sources in CR, Shield Open
"20220203_135029",
"20220203_144253",
"20220203_145146",
"20220203_145315",
"20220203_150210",
"20220203_150507",
"20220203_151401",
"20220203_152254",
"20220203_152257",
"20220203_152300",
"20220203_153153",
"20220203_161023"]

series_list_3 = [ ## Al resonator, No sources in CR, Shield Closed
# "20220223_154540",
# "20220223_163334",
# "20220223_172123",
# "20220223_180912",
# "20220223_185707",
# "20220223_194457",
# "20220223_203246",
# "20220223_212036",
# "20220223_220831",
# "20220223_225623",
# "20220302_174251",
# "20220302_183051",
# "20220302_191845",
# "20220302_200631",
# "20220302_205426",
"20220304_094854",
"20220317_094420",
"20220318_001948" # laser on
]

class_names = [
"Al, Ba+Cs sources",
"Al, Shield open",
"Al, Shield closed",
]

colors = ["g", "b", "r"]

series_bus = np.empty(3, dtype=object)
series_bus[0] = series_list_1
series_bus[1] = series_list_2
series_bus[2] = series_list_3

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

## Loop over each class of data
for j in np.arange(len(series_bus)):

	## Pull the list of series for this class
	series_list = series_bus[j]

	## Loop over each series in this class
	for i in np.arange(len(series_list)):

		fullpath = os.path.join(datapath,"out_"+series_list[i])
		filename = "ResonanceFits_"+series_list[i]+".h5"

		try:
			fdata = fit.decode_hdf5(os.path.join(fullpath,filename))
		except:
			print("Problem with file:",os.path.join(fullpath,filename))
			print("Skipping...")
			continue

		ax10.plot(fdata.powers, fdata.fit_fr, 
			alpha=0.5, color=colors[j], label=class_names[j] if i==0 else None)

		ax20.plot(fdata.powers, (fdata.fit_fr-np.mean(fdata.fit_fr))/fdata.fit_fr, 
			alpha=0.5, color=colors[j], label=class_names[j] if i==0 else None)

		ax30.plot(fdata.powers, fdata.fit_Qr,
			alpha=0.5, color=colors[j], label=class_names[j] if i==0 else None)

fig1.gca() ; ax10.legend(loc="best") ; plt.tight_layout() ; fig1.savefig("/home/nexus-admin/Downloads/Figure_1.png")
fig2.gca() ; ax20.legend(loc="best") ; plt.tight_layout() ; fig2.savefig("/home/nexus-admin/Downloads/Figure_2.png")
fig3.gca() ; ax30.legend(loc="best") ; plt.tight_layout() ; fig3.savefig("/home/nexus-admin/Downloads/Figure_3.png")
plt.show()

