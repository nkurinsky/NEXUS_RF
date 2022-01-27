import numpy as n
import h5py

class VNAMeas:

	## Class attributes
	date    = "20220101"			## [STRING] YYYYMMDD
	series  = "20220101_000000"		## [STRING] YYYYMMDD_HHMMSS

	power   = 0.0					## [FLOAT] (dBM) RF stimulus power
	n_avgs  = 1						## [INT] How many sweeps to take at a given power
	n_samps = 5e4					## [FLOAT] How many samples to take evenly spaced in freq range

	f_min   = 4.24205e9				## [FLOAT] (Hz) minimum frequency of sweep range
	f_max   = 4.24225e9				## [FLOAT] (Hz) minimum frequency of sweep range

	start_T = 0.0 					## [FLOAT] (mK) Temperature at start of the f sweep
	final_T = 0.0 					## [FLOAT] (mK) Temperature at end of the f sweep

	frequencies = np.array([])
	S21realvals = np.array([])
	S21imagvals = np.array([])

	def __init__(self, date_str, series):
		self.date   = date_str
		self.series = series
		return None

	def save_hdf5(self, filename):

		with h5py.File(filename+".h5", "w") as f:
			d_date    = f.create_dataset("date"   , data=self.date)
			d_series  = f.create_dataset("series" , data=self.series)

			d_power   = f.create_dataset("power"  , data=self.power)
			d_n_avgs  = f.create_dataset("n_avgs" , data=self.n_avgs)
			d_n_samps = f.create_dataset("n_samps", data=self.n_samps)

			d_f_min   = f.create_dataset("f_min"  , data=self.f_min)
			d_f_max   = f.create_dataset("f_max"  , data=self.f_max)

			d_start_T = f.create_dataset("start_T", data=self.start_T)
			d_final_T = f.create_dataset("final_T", data=self.final_T)

			d_frequencies = f.create_dataset("frequencies", data=self.frequencies)
			d_S21realvals = f.create_dataset("S21realvals", data=self.S21realvals)
			d_S21imagvals = f.create_dataset("S21imagvals", data=self.S21imagvals)

			f.close()
			
		return filename+".h5"


