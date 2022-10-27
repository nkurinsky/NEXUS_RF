import socket
from time import sleep

default_pulse_params = {
    "f_Hz" : 100.0,
    "pw_us":  10.0,
    "V_hi" :   5.0,
    "V_lo" :   0.0,
}

class AFG3102():

    def __init__(self,server_ip="192.168.0.142",server_port=1234,gpib_addr=11):
        self.address   = (server_ip, server_port)
        self.gpib_addr = gpib_addr

    ## Sends a command to the server address and returns an array of 
    ## strings containing the parts of the server response string
    ## between the commas
    def _sendCmd(self,cmd,getResponse=True,verbose=False):
        ## Append a newline character to the end of the line
        if not (cmd[-1]=="\n"):
            cmdStr = cmd+"\n"

        ## Diagnostic text
        if verbose:
            print("Sending command:",cmd,"to IP:",self.address)

        ## Open the socket and send/receive data
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.address)
                s.settimeout(1)
                s.sendall(cmdStr.encode())
                if(getResponse):
                    s.sendall("++read\n".encode())
                    data   = s.recv(1024)
                    retStr = data.decode()
                    if verbose:
                        print("Received:", retStr)
                else:
                    retStr = ""
        except socket.timeout:
            print("Timeout on", self.address[0])
            return

        ## Remove leading or trailing whitespace in string response
        ## as well as any quotation marks
        retStr   = retStr.strip().strip("\'").strip("\"")

        ## Return the split comma-separated response
        sleep(0.05)
        if (getResponse):
            return retStr.split(",")

    ## Checks to see if there's communication on the server address
    def testConnection(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(self.address)
            except ConnectionRefusedError:
                print("ERROR -- Connection refused by remote host")
            except OSError:
                print("ERROR -- No route to host, check IP address")
            except socket.timeout:
                print("Timeout on", self.server_address[0])
            else:
                print("Connection OK")
        return

    ## Call this once after instantiating the class
    def configureGPIB(self):
        ## Set mode as CONTROLLER
        self._sendCmd("++mode 1", getResponse=False)

        ## Turn off read-after-write to avoid "Query Unterminated" errors
        self._sendCmd("++auto 0", getResponse=False)

        ## Do not append CR or LF to GPIB data
        self._sendCmd("++eos 3", getResponse=False)

        ## Assert EOI with last byte to indicate end of data
        self._sendCmd("++eoi 1", getResponse=False)

        ## Read timeout is 500 msec
        self._sendCmd("++read_tmo_ms 500", getResponse=False)

        return

    ## Call this before sending any commands to ensure the GPIB-LAN interface
    ## is focusing on the correct instrument via its GPIB address
    def focusInstrument(self):
        ## Set Tek AFG3102 address
        self._sendCmd("++addr " + str(int(self.gpib_addr)), getResponse=False)
        return

    ## Get the standard Identity string of the device
    def getIdentity(self):
        resp = self._sendCmd("*IDN?")
        return resp ## array of strings

    ## Clear any errors on the device
    def clearErrors(self):
        self._sendCmd("*CLS", getResponse=False)
        return 
        
    ## Perform a soft reset of the device
    def doSoftReset(self):
        self._sendCmd("*RST", getResponse=False)
        return 

    ## Query if the output is currently on or off
    def getOutputState(self, ch=1):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return

        resp = self._sendCmd("OUTPut"+str(int(ch))+"?")
        return resp[0] ## "1" or "0"

    ## Set the output state (ON/OFF)
    def setOutputState(self, ch=1, enable=True, confirm=True):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return

        cmd  = "OUTPut" + str(int(ch))
        arg  = "ON" if enable else "OFF"
        cmd_str = " ".join( ( cmd, arg ) )
        self._sendCmd(cmd_str, getResponse=False)
        if confirm:
            print("Output is:", "ON" if bool(int(self.getOutputState())) else "OFF")
        return

    ## Update the frequency and derived parameters
    def updateFrequency(self, freq_Hz, ch=1, confirm=True, burst=True):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return
        ch_str = "SOURce" + str(int(ch))

        ## Extract the pulse parameter dictionary and create strings
        prd_sec = 1./freq_Hz
        Npulses = int(freq_Hz)

        f_str = "{:.3f}".format(freq_Hz) + "Hz"
        P_str = "{:.3f}".format(prd_sec) + "s"
        N_str = "{:d}".format(  Npulses)

        ## Send the commands to set up the source
        self._sendCmd(ch_str+":FREQuency:FIXed "+ f_str, getResponse=False)
        self._sendCmd(ch_str+":PULSe:PERiod "   + P_str, getResponse=False)
        if burst:
            self._sendCmd(ch_str+":BURSt:NCYCles "  + N_str, getResponse=False)

        ## Check the settings
        if confirm:
            print("Frequency [Hz]:", self._sendCmd(ch_str+":FREQuency:FIXed?") )
            print("Period   [sec]:", self._sendCmd(ch_str+":PULSe:PERiod?") )
            if burst:
                print("N cycles   [#]:", self._sendCmd(ch_str+":BURSt:NCYCles?") )

        return

    ## Update the low level voltage for the waveform
    def updateLoVoltage(self, V_lo, ch=1, confirm=True):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return
        ch_str = "SOURce" + str(int(ch))

        ## Send the commands to set up the source
        Vl_str = "{:.3f}".format(V_lo) + "V"
        self._sendCmd(ch_str+":VOLTage:LEVel:LOW "+Vl_str, getResponse=False)

        ## Check the settings
        if confirm:
            print("Voltage low  [V]:", self._sendCmd(ch_str+":VOLTage:LEVel:LOW?") )

        return

    ## Update the high level voltage for the waveform
    def updateHiVoltage(self, V_hi, ch=1, confirm=True):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return
        ch_str = "SOURce" + str(int(ch))

        ## Send the commands to set up the source
        Vh_str = "{:.3f}".format(V_hi) + "V"
        self._sendCmd(ch_str+":VOLTage:LEVel:HIGH "+Vh_str, getResponse=False)

        ## Check the settings
        if confirm:
            print("Voltage high [V]:", self._sendCmd(ch_str+":VOLTage:LEVel:HIGH?") )

        return

    ## Update the pulse width
    def updatePulseWidth(self, pw_us, ch=1, confirm=True):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return
        ch_str = "SOURce" + str(int(ch))

        ## Send the commands to set up the source
        pw_str = "{:.3f}".format(pw_us) + "us"
        self._sendCmd(ch_str+":PULSe:WIDTh "+pw_str, getResponse=False)

        ## Check the settings
        if confirm:
            print("Pulse width:", self._sendCmd(ch_str+":PULSe:WIDTh?") )

        return

    ## Set up a triggered output where the number of pulses in a single burst fills a full second
    def configureSource(self, pulse_par_dict, ch=1, confirm=True):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return
        ch_str = "SOURce" + str(int(ch))

        ## Send the commands to set up the source
        self._sendCmd(ch_str+":FUNCtion PULSe"      , getResponse=False)
        if confirm:
            print("Function   :", self._sendCmd(ch_str+":FUNCtion?"))
        self._sendCmd(ch_str+":BURSt:MODE TRIGgered", getResponse=False)
        if confirm:
            print("Burst  mode:", self._sendCmd(ch_str+":BURSt:MODE?"))
        self._sendCmd(ch_str+":FREQuency:MODE FIXed", getResponse=False)
        if confirm:
            print("Freq   mode:", self._sendCmd(ch_str+":FREQuency:MODE?"))

        self.updateHiVoltage( pulse_par_dict["V_hi" ], ch=ch, confirm=confirm )
        self.updateLoVoltage( pulse_par_dict["V_lo" ], ch=ch, confirm=confirm )
        self.updateFrequency( pulse_par_dict["f_Hz" ], ch=ch, confirm=confirm )
        self.updatePulseWidth(pulse_par_dict["pw_us"], ch=ch, confirm=confirm )

        self._sendCmd(ch_str+":BURSt:TDELay MINimum", getResponse=False)
        if confirm:
            print("Burst delay:", self._sendCmd(ch_str+":BURSt:TDELay?"))

        self._sendCmd(ch_str+":BURSt:STATe ON", getResponse=False)
        if confirm:
            print("Burst state:", self._sendCmd(ch_str+":BURSt:STATe?"))

        self._sendCmd("TRIGger:SEQuence:SOURce EXTernal")
        if confirm:
            print("Burst state:", self._sendCmd("TRIGger:SEQuence:SOURce"))

        # "SOURce1:FUNCtion PULSe"
        # "SOURce1:BURSt:MODE TRIGgered"
        # "SOURce1:FREQuency:MODE FIXed"

        # "SOURce1:VOLTage:LIMit:HIGH 1V" 5V
        # "SOURce1:VOLTage:LIMit:LOW 10mV" 0V
        # "SOURce1:FREQuency:FIXed 500kHz" 100Hz
        # "SOURce1:PULSe:PERiod 200ns" 1/100Hz
        # "SOURce1:BURSt:NCYCles 100" same as frequency
        # "SOURce1:PULSe:WIDTh 200ns" 10us

        # "SOURce1:BURSt:STATe ON"
        return

    ## Set up a continuous pulse output (no trigger)
    def configureContinuousSource(self, pulse_par_dict, ch=1, confirm=True):
        ## First check that the channel provided is okay
        if not (ch==1 or ch==2):
            print("Error:", ch, "is not a valid channel string. Options: 1, 2")
            return
        ch_str = "SOURce" + str(int(ch))

        ## Send the commands to set up the source
        self._sendCmd(ch_str+":FUNCtion PULSe"      , getResponse=False)
        if confirm:
            print("Function   :", self._sendCmd(ch_str+":FUNCtion?"))
        self._sendCmd(ch_str+":FREQuency:MODE FIXed", getResponse=False)
        if confirm:
            print("Freq   mode:", self._sendCmd(ch_str+":FREQuency:MODE?"))

        self.updateHiVoltage( pulse_par_dict["V_hi" ], ch=ch, confirm=confirm )
        self.updateLoVoltage( pulse_par_dict["V_lo" ], ch=ch, confirm=confirm )
        self.updateFrequency( pulse_par_dict["f_Hz" ], ch=ch, confirm=confirm, burst=False )
        self.updatePulseWidth(pulse_par_dict["pw_us"], ch=ch, confirm=confirm )

        self._sendCmd(ch_str+":BURSt:STATe OFF", getResponse=False)
        if confirm:
            print("Burst state:", self._sendCmd(ch_str+":BURSt:STATe?"))

        return

   
    