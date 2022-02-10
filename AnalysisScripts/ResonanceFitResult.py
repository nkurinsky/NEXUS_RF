import numpy as np
import h5py

class SinglePeakResult:

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
		print("Parameter estimates for peak", self.pk_idx, "(Added? "+str(self.pk_added)+")")
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

class SingleFileResult:

	## Class attributes
	in_fname = "Psweep_P-00.0_20220101_000000.h5"

	power   = 0.0					## [FLOAT] (dBM) RF stimulus power
	n_pks   = 1						## [INT] How many peaks were found in this file

	start_T = np.array([])			## [array of FLOAT] (mK) Temperature at start of the f sweep
	final_T = np.array([])			## [array of FLOAT] (mK) Temperature at end of the f sweep

	peak_fits = np.zeros(1, dtype=object)

	def __init__(self, file_name):
		self.in_fname   = file_name.split('/')[-1]
		return None

	def resize_peak_fits(self, n_peaks):
		self.n_pks = n_peaks
		self.peak_fits = np.zeros(len(self.n_pks), dtype=object)

	def show_metadata(self):
		print("In file:     ", self.in_fname)
		print("Power [dBm]: ", self.power)
		print("# of peaks:  ", self.n_pks)
		print("Start T [mK]:", self.start_T)
		print("Final T [mK]:", self.final_T)

	def show_fit_results(self):
		for i in range(self.n_pks):
			self.peak_fits[i].show_fine_result()

class SeriesFitResult:

	## Class attributes
	date    = "20220101"			## [STRING] YYYYMMDD
	series  = "20220101_000000"		## [STRING] YYYYMMDD_HHMMSS

	n_files = 1						## [INT] How many files are in this series

	fit_fr  = np.array([])			## [array of FLOAT] (Hz) Central frequency 
	fit_Qr  = np.array([])			## [array of FLOAT] Quality factor
	fit_Qi  = np.array([])			## [array of FLOAT] Quality factor
	fit_Qc  = np.array([])			## [array of FLOAT] Quality factor

	file_fits = np.zeros(1, dtype=object)

	def __init__(self, date, series):
		self.date   = date
		self.series = series
		return None

	def resize_file_fits(self, n_files):
		self.n_files   = n_files
		self.file_fits = np.zeros(len(self.n_files), dtype=object)
		self.fit_f     = np.zeros(len(self.n_files))
		self.fit_Q     = np.zeros(len(self.n_files))

	def show_series_result(self):
		for i in range(self.n_files):
			self.file_fits[i].show_metadata()
			self.file_fits[i].show_fit_results()