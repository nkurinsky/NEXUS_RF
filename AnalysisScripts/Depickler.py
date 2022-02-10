import os
import pickle
import ResonanceFitResult

## Series identifier
day    = '20220127'
time   = '160519'
series = day + '_' + time

out_path = '/data/ProcessedOutputs/out_' + series

file = open(os.path.join(out_path,"Psweep_FitResults.pkl"),'rb')
obj = pickle.load(file)
file.close()

##
print("Date:   ", obj.date)
print("Series: ", obj.series)
print("# files:", obj.n_files)

print("Fr (GHz):", obj.fit_fr)
print("Qr      :", obj.fit_Qr)
print("Qi      :", obj.fit_Qi)
print("Qc      :", obj.fit_Qc)

## 
for i in range(obj.n_files):
	obj.file_fits[i].show_metadata()
	obj.file_fits[i].show_fit_results()