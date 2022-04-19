import socket
from time import sleep

## Use this class to interface with the thermometer boxes (MGC3)
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
            print(vnames[i], ":", self._getVar(i))   

    def getTemp(self):
        try:
            ans = float(self._getVar(3)) ## K
        except socket.timeout:
            print("Timeout on", self.server_address[0])
            ans = -99.99
        return ans

    def getSP(self):
        try:
            ans = float(self._getVar(2)) ## K
        except socket.timeout:
            print("Timeout on", self.server_address[0])
            ans = -99.99
        return 

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
            print(vnames[i], ":", self._getVar(i))    

    def getResistance(self):
        try:
            ans = float(self._getVar(3)) ## Ohm?
        except socket.timeout:
            print("Timeout on", self.server_address[0])
            ans = -99.99
        return ans

