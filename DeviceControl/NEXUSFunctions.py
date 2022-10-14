import os
import socket
import datetime
import numpy as np
import pandas as pd
from time import sleep

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from   pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

## Use this class to interface with the thermometer boxes (MGC3)
## This is used for live polling/setting of parameters on the heaters
class NEXUSHeater:

    def __init__(self,server_ip="192.168.0.34",server_port=11034):
        self.server_address = (server_ip, server_port)

    def _sendCmd(self,cmd,getResponse=True,sleepTime=0.05):
        cmdStr = cmd+"\n"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(self.server_address)
            s.settimeout(1)
            s.sendall(cmdStr.encode())
            if(getResponse):
                data = s.recv(1024)
                retStr = data.decode()

        retSplit = retStr.split("\n")
        retVals = retSplit[1:-1]
        retFlags = retSplit[0]

        sleep(sleepTime)
        if (getResponse):
            return retFlags,retVals

    def testConnection(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(self.server_address)
            except ConnectionRefusedError:
                print("ERROR -- Connection refused by remote host")
            except OSError:
                print("ERROR -- No route to host, check IP address")
            else:
                print("Connection OK")

    def getVersion(self):
        f,v = self._sendCmd("2;8");
        return v[0];

    def _getAllVars(self):
        f,v = self._sendCmd("2;2")
        return v

    def _getVar(self,var):
        f,v = self._sendCmd("2;7;"+str(var))
        vStr = v[0]
        return vStr.split(";")[-1]

    def showAllVars(self):
        vnames = self._getAllVars()
        for i in range(len(vnames)):
            try: 
                print(vnames[i], ":", self._getVar(i))
            except socket.timeout:
                print(vnames[i], ":", "Timeout on", self.server_address[0])

    def getTemp(self):
        try:
            ans = float(self._getVar(3)) ## K
        except socket.timeout:
            print("Timeout on", self.server_address[0])
            ans = -99.99
        return ans

    def getSP(self):
        try:
            # ans = float(self._sendCmd("2;7;2"))
            ans = float(self._getVar(2)) ## K
        except socket.timeout:
            print("Timeout on", self.server_address[0])
            ans = -99.99
        return ans

    def setSP(self,sp,scale="K"):
        if(scale == "K"):            
            spf = float(sp)
        elif(scale == "mK"):
            spf = float(sp)*1e-3
            
        if(spf > 0.350):
            raise ValueError("Temperature too high, limit is 350 mK")
        elif(spf < 0.01):
            raise ValueError("Temperature too low, limit is 10 mK")

        print("Changing setpoint to "+str(spf))
        
        f,v = self._sendCmd("1;2;"+str(spf))
        return

## Use this class to interface with the thermometer boxes (MMR3)
## This is used for live polling/setting of parameters on the thermometers
class NEXUSThermometer:

    ## By default this connects to the second box, where channel 0 is MC
    def __init__(self,server_ip="192.168.0.32",server_port=11032):
        self.server_address = (server_ip, server_port)

    def _sendCmd(self,cmd,getResponse=True,sleepTime=0.05):
        cmdStr = cmd+"\n"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(self.server_address)
            s.settimeout(1)
            s.sendall(cmdStr.encode())
            if(getResponse):
                data = s.recv(1024)
                retStr = data.decode()

        retSplit = retStr.split("\n")
        retVals = retSplit[1:-1]
        retFlags = retSplit[0]

        sleep(sleepTime)
        if (getResponse):
            return retFlags,retVals

    def testConnection(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(self.server_address)
            except ConnectionRefusedError:
                print("ERROR -- Connection refused by remote host")
            except OSError:
                print("ERROR -- No route to host, check IP address")
            else:
                print("Connection OK")

    def getVersion(self):
        f,v = self._sendCmd("2;8");
        return v[0];

    def _getAllVars(self):
        f,v = self._sendCmd("2;2")
        return v

    def _getVar(self,var):
        f,v = self._sendCmd("2;7;"+str(var))
        vStr = v[0]
        return vStr.split(";")[-1]

    def showAllVars(self):
        vnames = self._getAllVars()
        for i in range(len(vnames)):
            try: 
                print(vnames[i], ":", self._getVar(i))
            except socket.timeout:
                print(vnames[i], ":", "Timeout on", self.server_address[0])

    def getResistance(self):
        try:
            ans = float(self._getVar(3)) ## Ohm?
        except socket.timeout:
            print("Timeout on", self.server_address[0])
            ans = -99.99
        return ans

    def getTemperature(self, ch=1):
        ans = -99.99
        idx = 0
        if ch==1:
            idx = 5
        if ch==2:
            idx = 16
        if ch==3:
            idx = 27
        if idx==0:
            return ans
            
        try:
            ans = float(self._getVar(idx)) ## K
        except socket.timeout:
            print("Timeout on", self.server_address[0])
        return ans


## Pull the actual data from the file
remote_path = "/gpfs/slac/staas/fs1/g/supercdms/www/nexus/fridge/files/"
local_path  = "/data/SlowDataLogCopies/"

def read_plclog_data(date_series, offset):
    ## The format for creating a date series is "%y%m%d"
    datalist=[]
    for iseries in date_series:
        data = pd.read_csv(os.path.join(local_path,iseries+".txt"), delimiter = '\t') 
        data['ctime'] = [ datetime.datetime.strptime( ed.strip(" ")+"-"+eh.strip(" ") , 
                                                      '%m/%d/%Y-%H:%M:%S') \
                          + offset for ed,eh in zip(data['date'],data['heures']) ]
        datalist.append(data)
    result = pd.concat(datalist)
    return result

def read_MACRT_data(date_series, offset):
    datalist=[]
    for iseries in date_series:
        data = pd.read_csv(os.path.join(local_path,"MACRT_"+iseries+".csv"), delimiter = ';') 
        data['ctime'] = [datetime.datetime.strptime(elem, '%m/%d/%Y %I:%M:%S %p')+offset for elem in data['Date']]
        datalist.append(data)
    result = pd.concat(datalist)
    return result

def read_Lakeshore_data(date_series, offset):
    datalist=[]
    for iseries in date_series:
        iseries = iseries.replace("-","")
        data = pd.read_csv(os.path.join(local_path,"Temperature_"+iseries+".txt"), delimiter = '\t') 
        # print(data.keys())
        data['ctime'] = [datetime.datetime.strptime(elem, '%Y-%m-%d %H:%M:%S.%f')+offset for elem in data['Time']]
        datalist.append(data)
    result = pd.concat(datalist)
    return result

def select_Lakeshore_channel(data_df, channel=2):
    df_out   = data_df.loc[data_df['Channel']==channel]
    ch_tps   = [s.replace("K","") for s in data_df['Temperature'].loc[data_df['Channel']==channel].values]
    ch_temps = [np.nan if s=='N/A' or s=='Out of Range' else float(s) for s in ch_tps]
    df_out['Temperature'] = ch_temps
    return df_out

## Give the date string and the number of days to create an array to read in 
## all required files
def create_date_range(date_str, num_days, fmt='%Y-%m-%d'):
     a = datetime.datetime.strptime(date_str, fmt)
     print('The starting date is: ')
     print(a)
     dateList = []
     for x in range (0, num_days):
         a_date = a + datetime.timedelta(days = x)
         dateList.append( a_date.strftime(fmt))
     return dateList

## Example polling and plotting of data
def poll_and_plot_plclog(date_str, num_days):
    series  = create_date_range(date_str, num_days, fmt="%y%m%d")
    offset  = datetime.timedelta(days=0, hours=0, minutes=0)
    data_df = read_plclog_data(series, offset)

    #Example of plotting
    f = plt.figure(figsize = (12,4))
    a = plt.gca()

    a.plot(data_df['ctime'], data_df['P1 mbar'], label='P1 mbar', color='dodgerblue')
    a.set_xlabel('Time')
    a.set_ylabel('Pressure [mbar]')

    plt.grid()
    plt.legend(loc="best")
    f.autofmt_xdate()
    myFmt = mdates.DateFormatter('%m-%d %H:%M:%S')
    a.xaxis.set_major_formatter(myFmt)
    return f

def poll_and_plot_MACRT(date_str, num_days):
    series  = create_date_range(date_str, num_days)
    offset  = datetime.timedelta(days=0, hours=0, minutes=0)
    data_df = read_MACRT_data(series, offset)
    
    #Example of plotting
    f = plt.figure(figsize = (12,4))
    a = plt.gca()

    a.plot(data_df['ctime'], data_df['MIXING CHAMB_Conv'], label='Mixing Chamber NR7', color='dodgerblue')
    a.set_xlabel('Time')
    a.set_ylabel('Temp [K]')

    plt.grid()
    plt.legend(loc="best")
    f.autofmt_xdate()
    myFmt = mdates.DateFormatter('%m-%d %H:%M:%S')
    a.xaxis.set_major_formatter(myFmt)
    return f

if __name__ == "__main__":
    f = poll_and_plot_MACRT('2022-06-18',11)
    a = f.gca()
    # a.set_title("Nexus Run 7 Cooldown Curve")