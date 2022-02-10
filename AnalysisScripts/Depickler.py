import pickle
import ResonanceFitResult

## Series identifier
day    = '20220127'
time   = '160519'
series = day + '_' + time

out_path = '/data/ProcessedOutputs/out_' + series

file = open(os.path.join(out_path,"Psweep_FitResults.pkl"),'rb')
object_file = pickle.load(file)
file.close()

object_file.show_series_result()