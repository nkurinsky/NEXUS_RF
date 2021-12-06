import serial,time

defPort="/dev/cu.usbmodemR32809502262"

class Attenuator:

    def __init__(self,portName=defPort,verbose=True,debug=False):
        try:
            self.port = serial.Serial(portName, baudrate=115200, timeout=0.0)
        except:
            raise(ValueError("Could not open serial port "+portName))
        self.verbose=verbose
        self.debug=debug
        if(verbose):
            self.getInfo()

    def send(self,cmd,sleepTime=0.1):
        if(self.debug):
            print("Sending: ",cmd)
        self.port.write((cmd+"\r\n").encode())
        time.sleep(sleepTime)
        rcv = self.port.readline()
        rcv = self.port.readline()
        rcv = self.port.readline()

        msg=list()
        while(len(rcv) > 0):
            rStr = rcv.decode()
            if(rStr[0] != '#'):
                msg.append(rStr)
                if(self.verbose):
                    print(rStr,end='')                
            rcv = self.port.readline()
        return msg

    def _checkChannel_(self,chan):
        if(type(chan) == int):
            if(chan == 1):
                return '1'
            elif(chan == 2):
                return '2'
            else:
                raise(ValueError("Invalid Channel "+str(chan)))
        elif(type(chan) == str):
            if((chan == '1') | (chan == '2')):
                return chan
            else:
                raise(ValueError("Invalid Channel "+chan))
        else:
            raise(TypeError("Channel type "+str(type(chan))+" not recognized"))

    def setOn(self,chan=None,atten=0):
        if(chan == None):
            self.setAll(atten)
        else:
            self.set(chan,atten)

    def setOff(self,chan=None,atten=95):
        if(chan == None):
            self.setAll(atten)
        else:
            self.set(chan,atten)
        
    def getStatus(self):
        retval = self.send("STATUS")
        attVals=dict()

        count=0
        for i in range(0,len(retval)):
            if(retval[i][0] == 'C'):
                if(count == 0):
                    attVals['CH1'] = retval[i][11:-2]
                    count+=1
                else:
                    attVals['CH2'] = retval[i][11:-2]
        return attVals

    def getInfo(self):
        return self.send("INFO")

    def getHelp(self):
        return self.send("HELP")

    def getDefault(self):
        return self.send("DEFAULT_ATTEN")

    def setDefault(self,atten=95.0):
        return self.send("DEFAULT_ATTEN "+str(atten))

    def set(self,channel,atten):
        return self.send("SET "+self._checkChannel_(channel)+" "+str(atten))

    def setAll(self,atten):
        return self.send("SAA "+str(atten))

    def setRandom(self,channel,low,high):
        return self.send("RAND "+self._checkChannel_(channel)," "+str(low)+" "+str(high))
