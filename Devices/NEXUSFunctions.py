import socket
from time import sleep

class NEXUSFridge:

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

    def getVersion(self):
        f,v = self._sendCmd("2;8");
        return v[0];

    def _getVar(self,var):
        f,v = self._sendCmd("2;7;"+str(var))
        vStr = v[0]
        return vStr[4:]

    def getTemp(self):
        return self._getVar(3)

    def getSP(self):
        return self._getVar(2)

    def setSP(self,sp,scale="K"):
        if(scale == "K"):            
            spf = float(sp)
        elif(scale == "mK"):
            spf = float(sp)*1e-3
            
        if(spf > 0.2):
            raise ValueError("Temperature too high, limit is 200 mK")
        elif(spf < 0.01):
            raise ValueError("Temperature too low, limit is 10 mK")

        print("Changing setpoint to "+str(spf))
        
        f,v = self._sendCmd("1;2;"+str(spf))
        return
