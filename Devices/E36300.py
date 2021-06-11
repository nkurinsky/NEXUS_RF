import socket

class E36300:

    def __init__(self,server_ip="192.168.0.40",server_port=5025):
        self.server_address = (server_ip, server_port)

    def _sendCmd(self,cmd,getResponse=True):
        cmdStr = cmd+"\n"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(self.server_address)
            s.settimeout(1)
            s.sendall(cmdStr.encode())
            if(getResponse):
                data = s.recv(1024)
                retStr = data.decode()

        if (getResponse):
            return retStr[:-1]

    def getID(self):
        return self._sendCmd("*IDN?");

    def parseChannel(self,channel):
        if(channel == None):
            return "(@1:3)"
        elif(type(channel) == list):
            retStr="(@"
            for i in range(0,len(channel)):
                if(i > 0):
                    retStr+=','
                retStr+=str(channel[i])
            retStr+=")"
            return retStr
        elif((channel < 1) | (channel > 3)):
            raise ValueError("Incorrect channel number")
        else:
            return "(@"+str(channel)+")"

    def parseResponse(self,response):
        strs = response.split(',')
        retvals = list()        
        for i in range(0,len(strs)):
            retvals.append(float(strs[i]))
        return retvals
    
    def getVoltage(self,channel=None):
        chStr=self.parseChannel(channel)
        return self.parseResponse(self._sendCmd("VOLT? "+chStr))

    def getStatus(self,channel=None):
        chStr=self.parseChannel(channel)
        retval = self._sendCmd("OUTP? "+chStr)
        strs = retval.split(',')
        retvals = list()
        for i in range(0,len(strs)):
            if(strs[i] == "0"):
                retvals.append('Off')
            else:
                retvals.append('On')
        return retvals
    
    def getCurrent(self,channel=None):
        if(channel != None):
            chStr=self.parseChannel(channel)
            return self.parseResponse(self._sendCmd("MEAS:CURR? "+chStr))
        else:
            currents = [0.0,0.0,0.0]
            for i in range(0,3):
                chStr=self.parseChannel(i+1)
                currents[i] = float(self._sendCmd("MEAS:CURR? "+chStr))
            return currents

    def setVoltage(self,channel=None,voltage=0.0):
        if(channel == None):
            #Don't adjust channels simultaneously
            raise ValueError("Please specify a channel, multiple voltages will not be set simultaneously")
        else:
            chStr=self.parseChannel(channel)
            self._sendCmd("VOLT "+str(voltage)+", "+chStr,False)

    def enable(self,channel=None):
        chStr=self.parseChannel(channel)
        self._sendCmd("OUTP ON, "+chStr,False)

    def disable(self,channel=None):
        chStr=self.parseChannel(channel)
        self._sendCmd("OUTP OFF, "+chStr,False)
