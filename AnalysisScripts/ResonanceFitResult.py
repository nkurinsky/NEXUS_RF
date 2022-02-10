import numpy as np
import h5py

class SinglePeakResult():

	pk_idx   = 0 		## For a given file, each peak found has a unique id
	pk_added = False	## Was the peak location added manually
	f_ctr    = 0.0 		## Identified central frequency of this peak
	mfz_ctr  = 0.0 		## Magnitude of filtered signal at central frequency of this peak

	## Parameter estimates
	f0_est = -1.0
	Qr_est = -1.0
	id_f0  = -1.0
	id_BW  = -1.0

	## Rough fit results
	rough_result = {"f0"    : -1.0, 
	                "Q"     : -1.0,
	                "phi"   : -1.0,
	                "zOff"  : -1.0,
	                "Qc"    : -1.0,
	                "tau1"  : -1.0,
	                "Imtau1": -1.0}

	## Fine fit parameter guesses
	fine_pguess = np.zeros(8)

	## Rough fit results
	fine_result = { "f0"    : -1.0, 
	                "Q"     : -1.0,
	                "phi"   : -1.0,
	                "zOff"  : -1.0,
	                "Qc"    : -1.0,
	                "tau1"  : -1.0,
	                "Imtau1": -1.0}
	fine_errors = { "f0"    : -1.0, 
	                "Q"     : -1.0,
	                "phi"   : -1.0,
	                "zOff"  : -1.0,
	                "Qc"    : -1.0,
	                "tau1"  : -1.0,
	                "Imtau1": -1.0}

	def __init__(self, peak_idx):
		self.pk_idx = peak_idx
		return None

	def show_par_ests(self):
		print("Parameter estimates for peak", self.pk_idx)
		print("f0_est", self.f0_est)
		print("Qr_est", self.Qr_est)
		print("id_f0" , self.id_f0)
		print("id_BW" , self.id_BW)

	def show_rough_result(self):
		print("Rough Fit result for peak", self.pk_idx)
		for key in self.rough_result.keys():
			print(key , ":" , self.rough_result[key] )

	def show_fine_result(self):
		print("Fine Fit result for peak",self.pk_idx)
		for key in self.fine_result.keys():
			print(key , ":" , self.fine_result[key] , "+/-" , self.fine_errors[key] )


class ResonanceFitResult:

	## Class attributes
	date    = "20220101"			## [STRING] YYYYMMDD
	series  = "20220101_000000"		## [STRING] YYYYMMDD_HHMMSS

	power   = 0.0					## [FLOAT] (dBM) RF stimulus power
	n_avgs  = 1						## [INT] How many sweeps to take at a given power
	n_samps = 5e4					## [FLOAT] How many samples to take evenly spaced in freq range

	f_min   = 4.24205e9				## [FLOAT] (Hz) minimum frequency of sweep range
	f_max   = 4.24225e9				## [FLOAT] (Hz) minimum frequency of sweep range

	start_T = np.array([])			## [array of FLOAT] (mK) Temperature at start of the f sweep
	final_T = np.array([])			## [array of FLOAT] (mK) Temperature at end of the f sweep

	frequencies = np.array([])
	S21realvals = np.array([])
	S21imagvals = np.array([])

	def __init__(self, date_str, series):
		self.date   = date_str
		self.series = series
		return None