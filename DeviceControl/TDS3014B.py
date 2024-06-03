import socket
from time import sleep

class TDS3014B():

    def __init__(self,server_ip="192.168.0.142",server_port=1234,gpib_addr=21):
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


   
    
