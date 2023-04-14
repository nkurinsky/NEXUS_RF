import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_average(t, a, n=3):
    ret     = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return t[int(n/2):-int(n/2-1)], ret[n - 1:] / n

def get_column(series,channel,colname='Resistance',datapath="/data/SlowDataLogCopies",offset_hrs=-5):
    ## Check col names 'Resistance' or 'Temperature'
    if (colname != 'Resistance') and (colname !='Temperature'):
        print("Bad column name:",colname)
        return None, None

    ## Find the file
    fname    = "Temperature_" + series + ".txt"
    filepath = os.path.join(datapath,fname)

    ## Pull the data and define a mask for pulling only interesting data
    data = []
    with open(filepath,"r") as f:
        data = pd.read_csv(f,"\t",header=0)
    mask = (data['Channel'] == channel) & [True if re.search('[0-9]*\.?[0-9]', x) else False for x in data[colname]]

    ## Pull the data defined by our mask, apply an offset from UTC to C(D/S)T
    offset = datetime.timedelta(days=0, hours=offset_hrs, minutes=0)
    data_vals = [float(i[0:-1]) for i in data[colname][mask]]
    time_vals = [datetime.datetime.strptime(elem, '%Y-%m-%d %H:%M:%S.%f')+offset for elem in data['Time'][mask]]
    
    return np.array(time_vals),np.array(data_vals)

if __name__ == "__main__":
    Time,Temperature = get_column("20230413",5,colname="Temperature")

    mask = (Time < datetime.datetime(2023, 4, 13, 9, 50, 14, 425000))
    plt.plot(Time[::1][mask],Temperature[::1][mask],"-o")

    plt.legend(fontsize=20)
    plt.xticks(rotation=45);
    plt.grid(which="both")
    plt.title("Thermomoter Temperature")
    plt.ylabel("Temperature [K]")
    plt.xlabel("Time")
    plt.show()