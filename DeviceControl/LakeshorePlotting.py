import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def LakeShoreTemp(series,channel):
    path = "/gpfs/slac/staas/fs1/g/supercdms/www/nexus/fridge/LakeshoreBridge/" + "Temperature_"

    data = []
    filename = path + series + ".txt"
    with open(filename,"r") as f:
        data = pd.read_csv(f,"\t",header=0)
    mask = (data['Channel'] == channel) & [True if re.search('[0-9]*\.?[0-9]', x) else False for x in data['Temperature']]

    offset = datetime.timedelta(days=0, hours=-5, minutes=0)
    Temperature = [float(i[0:-1]) for i in data['Temperature'][mask]]
    Time = [datetime.datetime.strptime(elem, '%Y-%m-%d %H:%M:%S.%f')+offset for elem in data['Time'][mask]]
    
    return np.array(Time),np.array(Temperature)

def moving_average(t, a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return t[int(n/2):-int(n/2-1)], ret[n - 1:] / n

Time,Temperature = LakeShoreTemp("20230413",5)

mask = (Time < datetime.datetime(2023, 4, 13, 9, 50, 14, 425000))
plt.plot(Time[::1][mask],Temperature[::1][mask],"-o")

#rolling_n = 5
#rolling_Time,rolling_Temp = moving_average(Time,Temperature,rolling_n)
#plt.plot(rolling_Time,rolling_Temp,linewidth=5,label="rolling ave")

plt.legend(fontsize=20)
plt.xticks(rotation=45);
plt.grid(which="both")
plt.title("Thermomoter Temperature")
plt.ylabel("Temperature [K]")
plt.xlabel("Time")