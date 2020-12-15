import socket

class ADR:

    def __init__(self,server_ip='ppd-124687.dhcp.fnal.gov',server_port=2060):
        self.server_address = (server_ip, server_port)

    def _sendCmd(self,cmd):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(self.server_address)
            s.sendall(cmd.encode())
            data = s.recv(1024)

        return data.decode()

    def getTemp(self):
        return float(self._sendCmd("therm?,")[:-2])

    def getSP(self):
        return float(self._sendCmd("setp?,")[:-2])

    def rampOff(self):
        retval=self._sendCmd("rampoff,")
        return retval
    
    def setSP(self,setPoint):
        #check for ramp status, turn off ramp if on
        rStat = self.getRamp()
        if(rStat[0]):
            self.rampOff()
        
        retval=self._sendCmd("setpt,"+str(round(setPoint,3)))
        return retval

    def getRamp(self):
        ramp = self._sendCmd("ramp?,")
        if(ramp[0] == "0"):
            ramp1 = False
        else:
            ramp1 = True

        if(ramp[3] == "0"):
            ramp2 = False
        else:
            ramp2 = True

        return [ramp1,ramp2]
