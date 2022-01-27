import numpy as np
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

	start_T = np.array([])			## [array of FLOAT] (mK) Temperature at start of the f sweep
	final_T = np.array([])			## [array of FLOAT] (mK) Temperature at end of the f sweep

	frequencies = np.array([])
	S21realvals = np.array([])
	S21imagvals = np.array([])

	def __init__(self, date_str, series):
		self.date   = date_str
		self.series = series
		return None

	def show(self):
		print("VNA Measurement:", self.series)
		print("====---------------------------====")
		print("|             Date:  ", self.date)
		print("|   RF Power [dBm]:  ", self.power)
		print("|   N averages:      ", self.n_avgs)
		print("|   N sweep samples: ", self.n_samps)
		print("|   Sweep f min [Hz]:", self.f_min)
		print("|   Sweep f max [Hz]:", self.f_max)
		for i in range(len(self.start_T)):
			print("|  Start Temp "+str(i)+ " [mK]:", self.start_T[i])
		for i in range(len(self.final_T)):
			print("|  Final Temp "+str(i)+ " [mK]:", self.final_T[i])
		print("|   # freq saved:    ", len(self.frequencies))
		print("|   # Re(S21) saved: ", len(self.S21realvals))
		print("|   # Im(S21) saved: ", len(self.S21imagvals))
		print("====---------------------------====")


	def save_hdf5(self, filename):

		with h5py.File(filename+".h5", "w") as f:
			d_date    = f.create_dataset("date"   , data=np.array([self.date], dtype='S'))
			d_series  = f.create_dataset("series" , data=np.array([self.series], dtype='S'))

			d_power   = f.create_dataset("power"  , data=np.array([self.power]))
			d_n_avgs  = f.create_dataset("n_avgs" , data=np.array([self.n_avgs]))
			d_n_samps = f.create_dataset("n_samps", data=np.array([self.n_samps]))

			d_f_min   = f.create_dataset("f_min"  , data=np.array([self.f_min]))
			d_f_max   = f.create_dataset("f_max"  , data=np.array([self.f_max]))

			d_start_T = f.create_dataset("start_T", data=self.start_T)
			d_final_T = f.create_dataset("final_T", data=self.final_T)

			d_frequencies = f.create_dataset("frequencies", data=self.frequencies)
			d_S21realvals = f.create_dataset("S21realvals", data=self.S21realvals)
			d_S21imagvals = f.create_dataset("S21imagvals", data=self.S21imagvals)

			f.close()

		return filename+".h5"


def decode_hdf5(filename):

	with h5py.File(filename, "r") as f:
		_date = f["date"][0].decode('UTF-8')
		_sers = f["series"][0].decode('UTF-8')
		sweep = VNAMeas(_date,_sers)

		sweep.power   = f["power"][0]
		sweep.n_avgs  = f["n_avgs"][0]
		sweep.n_samps = f["n_samps"][0]
		sweep.f_min   = f["f_min"][0]
		sweep.f_max   = f["f_max"][0]

		sweep.start_T = np.array(f["start_T"])
		sweep.final_T = np.array(f["final_T"])
		sweep.frequencies = np.array(f["frequencies"])
		sweep.S21realvals = np.array(f["S21realvals"])
		sweep.S21imagvals = np.array(f["S21imagvals"])

		return sweep