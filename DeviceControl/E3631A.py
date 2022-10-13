import socket
from time import sleep

class E3631A():

    def __init__(self,server_ip="192.168.0.142",server_port=1234,gpib_addr=5):
        self.address   = (server_ip, server_port)
        self.gpib_addr = gpib_addr

    ## Sends a command to the server address and returns an array of 
    ## strings containing the parts of the server response string
    ## between the commas
    def _sendCmd(self,cmd,getResponse=True,verbose=False):
        ## Append a newline character to the end of the line
        if not (cmdStr[-1]=="\n"):
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
                    data   = s.recv(1024)
                    retStr = data.decode()
                    if verbose:
                        print("Received:", retStr)
        except socket.timeout:
            print("Timeout on", self.server_address[0])
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
        ## Set HP E3631A address
        self._sendCmd("++addr " + str(int(self.gpib_addr)), getResponse=False)
        return

    ## Get the standard Identity string of the device
    def getIdentity(self):
        resp = self._sendCmd("*IDN?")
        return resp ## array of strings
        
    ## Perform a soft reset of the device
    def doSoftReset(self):
        self._sendCmd("*RST", getResponse=False)
        return 

    ## Query if the output is currently on or off
    def getOutputState(self):
        resp = self._sendCmd("OUTPut?")
        return resp[0] ## "1" or "0"

    ## Set the output state
    def setOutputState(self, enable=True, confirm=True):
        cmd  = "OUTPut"
        arg  = "ON" if enable else "OFF"
        cmd_str = " ".join( ( cmd, arg ) )
        self._sendCmd(cmd_str, getResponse=False)
        if confirm:
            print("Output is:", "ON" if (self.getOutputState()=="1") else "OFF")
        return

    ## Get the current output settings
    def getVoltage(self, ch="P6V"):
        ## First check that the channel provided is okay
        if not (ch=="P6V" or ch=="P25V" or ch=="N25V"):
            print("Error:", ch, "is not a valid channel string. Options: P6V, P25V, N25V")
            return

        resp = self._sendCmd(cmd+"? "+ch)
        return resp

    ## Apply a voltage/current on a specific output
    def setVoltage(self, voltage, current_limit=1.0, ch="P6V", confirm=True):

        ## First check that the channel provided is okay
        if not (ch=="P6V" or ch=="P25V" or ch=="N25V"):
            print("Error:", ch, "is not a valid channel string. Options: P6V, P25V, N25V")
            return

        ## Parse the voltage and current values
        vlt_str = "{:.3f}".format(voltage)
        cur_str = "{:.3f}".format(current_limit)
        if ( ch=="P6V" ):
            if voltage < 0.0:
                vlt_str = "MIN"
            if voltage > 6.0:
                vlt_str = "MAX"
            if current < 0.0:
                cur_str = "MIN"
            if current > 5.0:
                cur_str = "MAX"

        if ( ch=="P25V" ):
            if voltage < 0.0:
                vlt_str = "MIN"
            if voltage > 25.0:
                vlt_str = "MAX"
            if current < 0.0:
                cur_str = "MIN"
            if current > 1.0:
                cur_str = "MAX"

        ## This may not work as expected depending on if the firmware
        ## expects a negative sign in the command
        if ( ch=="N25V" ):
            if voltage > 0.0:
                vlt_str = "MIN"
            if voltage < -25.0:
                vlt_str = "MAX"
            if current < 0.0:
                cur_str = "MIN"
            if current > 1.0:
                cur_str = "MAX"

        ## Define the command string to send
        cmd  = "APPLy"
        args = ( ch, vlt_str, cur_str )
        cmd_str = " ".join( ( cmd, ",".join(args) ) )

        ## If we're on the right channel, apply the voltage
        self._sendCmd(cmd_str, getResponse=False)

        if confirm:
            resp = self.getVoltage(ch=ch)
            print("Settings:", resp)

    def measureVoltage(self, ch="P6V"):
        ## First check that the channel provided is okay
        if not (ch=="P6V" or ch=="P25V" or ch=="N25V"):
            print("Error:", ch, "is not a valid channel string. Options: P6V, P25V, N25V")
            return

        resp = self._sendCmd("MEASure:VOLTage? "+ch)
        return resp

    def measureCurrent(self, ch="P6V"):
        ## First check that the channel provided is okay
        if not (ch=="P6V" or ch=="P25V" or ch=="N25V"):
            print("Error:", ch, "is not a valid channel string. Options: P6V, P25V, N25V")
            return

        resp = self._sendCmd("MEASure:CURRent? "+ch)
        return resp