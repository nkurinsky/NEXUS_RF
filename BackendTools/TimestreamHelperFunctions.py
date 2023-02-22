import os, sys, glob
import time, datetime
import h5py
import numpy as np
import matplotlib.pyplot as plt

import PyMKID_USRP_functions as PUf
import PyMKID_resolution_functions as Prf

def logMag(z):
	return np.log10( np.abs(z) )

def phase(z):
	return np.angle(z)

def timesFromRate(rate, length):
	## rate passed in "samples per second"
	## length is the number of points in array
	f = 1./rate
	times = np.arange(start=0, stop=f*(length), step=f)
	return times

## Find all the uncleaned files that correspond to a specific timestream acquisition
def GetFiles(series, verbose=False, base_path='/data/USRP_Noise_Scans', sep_noise_laser=False):
	data_path = os.path.join(base_path,series.split('_')[0],series)
	
	## Grab the file with no calibration offset
	file_list = np.sort(glob.glob(data_path+"/*.h5"))

	s_file  = "None"
	d_file  = "None"
	v_file  = "None"
	n_files = []
	l_files = []

	for file in file_list:

		fname = file.split("/")[-1]

		if "USRP_Delay" in fname:
			d_file = file

		if "USRP_VNA" in fname:
			v_file = file

		if "noise_averages" in fname:
			s_file = file

		if "USRP_Noise" in fname and not "cleaned" in fname:
			n_files = np.append(n_files, file)

		if "USRP_Laser" in fname and not "cleaned" in fname:
			l_files = np.append(l_files, file)

	if verbose:
		print("Line Delay file: ",d_file.split("/")[-1] )
		print("VNA scan file:   ",v_file.split("/")[-1] )
		print("Noise ts files:  ",[ file.split("/")[-1] for file in n_files ])
		print("Laser ts files:  ",[ file.split("/")[-1] for file in l_files ])
		print("Summary file:	",s_file.split("/")[-1] )

	if sep_noise_laser:
		return s_file, d_file, v_file, n_files, l_files
		
	t_files = np.append(n_files, l_files)
	return s_file, d_file, v_file, t_files

## Pull metadata from the noise averages summary file, store in dictionary
def UnpackSummary(s_file_path, verbose=False):
	
	## Open the summary file in read-only mode
	fsum = h5py.File(s_file_path, 'r')
	if verbose:
		print("File keys:            ", fsum.keys())
		print("File attribute keys:  ", fsum.attrs.keys())
		
	## Should only be one power per file
	## Open that data member
	md = None
	try:
		md   = fsum['Power0']
	except:
		for i in np.arange(20):
			try:
				md   = fsum['Power'+str(i)]
			except:
				continue
	if md is None:
		print("Error - Cannot find appropriate power data group.")
		return None, None, None
	if verbose:
		print("PowerN keys:          ", md.keys())
		print("PowerN attribute keys:", md.attrs.keys())
		
	## Create a dictionary to store results
	md_dict = {}
	
	## Pull the metadata from the file into a dictionary
	for k in md.attrs.keys():
		md_dict[k] = md.attrs[k]

	## If there is laser data on top of the attributes add it to the dictionary
	for k in md.keys():
		# if "LaserScan" in k:
		if "Scan" in k:
			l_dict = {}
			for kk in md[k].keys():
				l_dict[kk] = md[k][kk].value
			md_dict[k] = l_dict

	## Pull the mean F,S21 from the cal delta scans
	mean_frqs = np.copy(np.array(md['freqs']))
	mean_S21s = np.copy(np.array(md['means']))
	
	## Close the summary h5 file
	fsum.close()

	return md_dict, mean_frqs, mean_S21s

def CleanPSDs(ts_file, vna_file, series=None, PSD_lo_f=1e2, PSD_hi_f=5e4, f_transient=0.3, charZs=None, charFs=None, MBresults=None, i=None, show_plots=True, verbose=True):
	
	if series is not None:
		sum_file, dly_file, vna_file, tone_files = GetFiles(series, verbose=verbose>1)
		ts_file = tone_files[0]

		metadata, charFs, charZs = UnpackSummary(sum_file, verbose=verbose>1)

	PSD_lo_f = int(PSD_lo_f)  ## chunk up to [Hz]
	PSD_hi_f = int(PSD_hi_f)  ## decimate down to  [Hz]
	
	if (f_transient > 1.0):
		f_transient = 0.3

	_, noise_info = PUf.unavg_noi(ts_file)
	noise_total_time = noise_info['time'][-1]
	noise_fs = 1./noise_info['sampling period']
	noise_readout_f = noise_info['search freqs'][0]

	num_chunks = int(noise_total_time*PSD_lo_f)
	noise_decimation = int(noise_fs/PSD_hi_f)

	if verbose>0:
		print("Will separate data into ", num_chunks	  , "chunks to achieve the requested", "{:.2e}".format(PSD_lo_f),' Hz low  end of the PSD')
		print("Additional decimation by", noise_decimation, "needed to achieve the requested", "{:.2e}".format(PSD_hi_f),' Hz high end of the PSD')

	p, P, r, t = Prf.PSDs_and_cleaning(ts_file, vna_file,
									  extra_dec  = noise_decimation,
									  num_chunks = num_chunks,
									  blank_chunks = int(f_transient*num_chunks),
									  removal_decimation = 1,
									  char_zs = charZs,
									  char_fs = charFs,
									  MB_results = MBresults,
									  i=i,
									  show_plots = show_plots,
									  verbose = verbose>0)
	return p, P, r, t ## powers, PSDs, res, timestreams

def PlotPSDsByPower(series_list, powers_list, fHz_range = [1e2,3e5],
	e_b_PSDrange = [1e-13,1e-10], r_b_PSDrange = [1e-21,1e-15],
	q_b_PSDrange = [1e-4,5e1], MB_fit_result=None,
	PSD_lo_f=1e2, PSD_hi_f=5e4, f_transient=0.3, show_sub_plots=False, verbose=False,
	f_data = [None], psd_to_file=False ):

	## Determine how many powers we'll loop over
	n_pwrs = len(series_list)

	## Create a container for the data at the requested frequencies
	if f_data[0] is not None:
		p_dict = {}
		n_frqs = len(f_data)
		n_vals = 6 if (MB_fit_result is not None) else 4

	## Create some file names, delete the files if they already exist
	if psd_to_file:
		f_psd_radius = "PSD_data_radius.csv"
		if os.path.exists(f_psd_radius):
			os.remove(f_psd_radius)
		f_psd_arclen = "PSD_data_arclength.csv"
		if os.path.exists(f_psd_arclen):
			os.remove(f_psd_arclen)
		f_psd_dissip = "PSD_data_dissipation.csv"
		if os.path.exists(f_psd_dissip):
			os.remove(f_psd_dissip)
		f_psd_freqcy = "PSD_data_frequency.csv"
		if os.path.exists(f_psd_freqcy):
			os.remove(f_psd_freqcy)
		if (MB_fit_result is not None):
			f_psd_kappa1 = "PSD_data_kappa1.csv"
			if os.path.exists(f_psd_kappa1):
				os.remove(f_psd_kappa1)
			f_psd_kappa2 = "PSD_data_kappa2.csv"
			if os.path.exists(f_psd_kappa2):
				os.remove(f_psd_kappa2)
		

	## Create the axes
	fga = plt.figure()
	axa = fga.gca()

	axa.set_xlabel("Frequency [Hz]")
	axa.set_ylabel(r"Radius PSD (Cleaned) [(ADCu)$^2$/Hz]")
	axa.set_xlim(fHz_range)
	axa.set_ylim(e_b_PSDrange)
	axa.set_xscale('log')
	axa.set_yscale('log')

	fgb = plt.figure()
	axb = fgb.gca()

	axb.set_xlabel("Frequency [Hz]")
	axb.set_ylabel(r"Arc Length PSD (Cleaned) [(ADCu)$^2$/Hz]")
	axb.set_xlim(fHz_range)
	axb.set_ylim(e_b_PSDrange)
	axb.set_xscale('log')
	axb.set_yscale('log')

	fgA = plt.figure()
	axA = fgA.gca()

	axA.set_xlabel("Frequency [Hz]")
	axA.set_ylabel(r"Dissipation PSD (Cleaned) [$(\delta(1/Q))^2$/Hz]")
	axA.set_xlim(fHz_range)
	axA.set_ylim(r_b_PSDrange)
	axA.set_xscale('log')
	axA.set_yscale('log')

	fgB = plt.figure()
	axB = fgB.gca()

	axB.set_xlabel("Frequency [Hz]")
	axB.set_ylabel(r"Frequency PSD (Cleaned) [$(\delta f/f)^2$/Hz]")
	axB.set_xlim(fHz_range)
	axB.set_ylim(r_b_PSDrange)
	axB.set_xscale('log')
	axB.set_yscale('log')

	if (MB_fit_result is not None):
		fg1 = plt.figure()
		ax1 = fg1.gca()

		ax1.set_title(r"Dissipation direction ($\kappa_1$)")
		ax1.set_xlabel("Frequency [Hz]")
		ax1.set_ylabel(r"$\kappa_1$ PSD (Cleaned) [$(\mu$m$^{-3})^2$/Hz]")
		ax1.set_xlim(fHz_range)
		ax1.set_ylim(q_b_PSDrange)
		ax1.set_xscale('log')
		ax1.set_yscale('log')

		fg2 = plt.figure()
		ax2 = fg2.gca()

		ax2.set_title(r"Frequency direction ($\kappa_2$)")
		ax2.set_xlabel("Frequency [Hz]")
		ax2.set_ylabel(r"$\kappa_2$ PSD (Cleaned) [$(\mu$m$^{-3})^2$/Hz]")
		ax2.set_xlim(fHz_range)
		ax2.set_ylim(q_b_PSDrange)
		ax2.set_xscale('log')
		ax2.set_yscale('log')

	## Loop over every series
	for i in np.arange(n_pwrs):
		sum_file, dly_file, vna_file, tone_files = GetFiles(series_list[i], verbose=verbose)
		
		metadata, avg_frqs, avg_S21s = UnpackSummary(sum_file, verbose=verbose)
		
		powers, PSDs, res, timestreams = CleanPSDs(tone_files[0], vna_file,
											   charFs      = avg_frqs,
											   charZs      = avg_S21s,
											   MBresults   = MB_fit_result,
											   PSD_lo_f    = PSD_lo_f, 
											   PSD_hi_f    = PSD_hi_f, 
											   f_transient = f_transient,
											   show_plots  = show_sub_plots,
											   verbose     = verbose )
		
		axa.plot(PSDs["f"],PSDs['radius'][:,0],label=str(powers_list[i])+" dBm")
		axb.plot(PSDs["f"],PSDs['arc'][:,0],label=str(powers_list[i])+" dBm")
		axA.plot(PSDs["f"],PSDs['dissipation'],label=str(powers_list[i])+" dBm")
		axB.plot(PSDs["f"],PSDs['frequency'],label=str(powers_list[i])+" dBm")
		if (MB_fit_result is not None):
			ax1.plot(PSDs["f"],PSDs['kappa_1'],label=str(powers_list[i])+" dBm")
			ax2.plot(PSDs["f"],PSDs['kappa_2'],label=str(powers_list[i])+" dBm")


		if psd_to_file:
			## Write the frequency line on the first pass only
			if i ==0
				h_line = "feedline_power," + ",".join(np.array(PSDs["f"]).astype('str')) + "\n"
				with open(f_psd_radius, "a") as file:
					file.write(h_line)
				with open(f_psd_arclen, "a") as file:
					file.write(h_line) 
				with open(f_psd_dissip, "a") as file:
					file.write(h_line)
				with open(f_psd_freqcy, "a") as file:
					file.write(h_line) 
				if (MB_fit_result is not None):
					with open(f_psd_kappa1, "a") as file:
						file.write(h_line) 
					with open(f_psd_kappa2, "a") as file:
						file.write(h_line) 

			## Now write the data line for this power
			with open(f_psd_radius, "a") as file:
				file.write(str(powers_list[i]) + "," + ",".join(np.array(PSDs["radius"]).astype('str')) + "\n")
			with open(f_psd_arclen, "a") as file:
				file.write(str(powers_list[i]) + "," + ",".join(np.array(PSDs["arc"]).astype('str')) + "\n")
			with open(f_psd_dissip, "a") as file:
				file.write(str(powers_list[i]) + "," + ",".join(np.array(PSDs["dissipation"]).astype('str')) + "\n")
			with open(f_psd_freqcy, "a") as file:
				file.write(str(powers_list[i]) + "," + ",".join(np.array(PSDs["frequency"]).astype('str')) + "\n")
			if (MB_fit_result is not None):
				with open(f_psd_kappa1, "a") as file:
					file.write(str(powers_list[i]) + "," + ",".join(np.array(PSDs["kappa_1"]).astype('str')) + "\n")
				with open(f_psd_kappa2, "a") as file:
					file.write(str(powers_list[i]) + "," + ",".join(np.array(PSDs["kappa_2"]).astype('str')) + "\n")

		if f_data[0] is not None:
			p_data  = np.zeros(shape=(n_frqs,n_vals+1))
			for jj in np.arange(n_frqs):
				## Find the closest PSD point to the requested frequency
				idx          = np.argmin(np.abs(PSDs["f"]-f_data[jj]))
				p_data[jj,0] = PSDs["f"][idx]

				p_data[jj,1] = PSDs['radius'][idx,0]
				p_data[jj,2] = PSDs['arc'][idx,0]
				p_data[jj,3] = PSDs['dissipation'][idx]
				p_data[jj,4] = PSDs['frequency'][idx]
				if (MB_fit_result is not None):
					p_data[jj,5] = PSDs['kappa_1'][idx]
					p_data[jj,6] = PSDs['kappa_2'][idx]

			p_dict[powers_list[i]] = p_data
			del p_data

		del sum_file, dly_file, vna_file, tone_files
		del metadata, avg_frqs, avg_S21s
		del powers, res, timestreams
		
	axa.legend(loc='lower right')
	axb.legend(loc='lower right')
	axA.legend(loc='lower right')
	axB.legend(loc='lower right')
	if (MB_fit_result is not None):
		ax1.legend(loc='lower right')
		ax2.legend(loc='lower right')

	# plt.show()
	return p_dict
