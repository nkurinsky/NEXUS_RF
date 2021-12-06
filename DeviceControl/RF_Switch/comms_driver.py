from urllib.request import urlopen
import sys

class IPInstrument:

    error = "None"

    def __init__(self, ip_addr):

        if (len(ip_addr.split("."))==4):
            self.ip_addr = ip_addr
        else:
            self.ip_addr = None

    def HTTPSendCommandGetResponse(self, cmd_to_send, verbose=False):

        # Specify the IP address of the switch box
        full_msg = "http://" + self.ip_addr + "/:" + cmd_to_send

        # Show the user the full message if they want
        if (verbose):
            print("Preparing to send message:", full_msg)

        # Send the HTTP command and try to read the result
        try:
            http_result = urlopen(full_msg, timeout=1)
            msg_return  = http_result.read()

            # Show the user the full response if they want
            if (verbose):
                print("Response from message:", msg_return)

            # The switch displays a web GUI for unrecognised commands
            if len(msg_return) > 100:
                self.error = "Error, command not found:"+cmd_to_send
                print(self.error)
                msg_return = "Invalid Command!"

        # Catch an exception if URL is incorrect (incorrect IP or disconnected)
        except:
            self.error = "Error, no response from device; check IP address and connections."
            print(self.error)
            msg_return = "No Response!"
            # sys.exit()      # Exit the script

        # Return the response
        return msg_return

    def TestConnection(self):
        model = self.HTTPSendCommandGetResponse("MN?")       # Get model name
        serln = self.HTTPSendCommandGetResponse("SN?")       # Get serial number

        return model, serln

    def SetSwitchState(self, switch_id, out_port):

        # Check that we're only acting on switch "A" or "B"
        if not ( (switch_id=="A") or (switch_id=="B") ):
            self.error = "Error, unregonized switch ID:"+switch_id
            print(self.error)
            return True ## error state

        # Check that the out port is either 1 (left) or 2 (right)
        if not ( (out_port==1) or (out_port==2) ):
            self.error = "Error, unregonized port #:"+str(out_port)
            print(self.error)
            return True ## error state

        # The command expects a 0 or 1
        out_port = int(out_port - 1)

        # If we've made it this far our message is OK to send
        msg    = "SET"+switch_id+"="+str(out_port)
        status = self.HTTPSendCommandGetResponse(msg, verbose=True)

        # Now query the state to ensure it worked
        resp  = self.HTTPSendCommandGetResponse(switch_id+"SWPORT?", verbose=True)

        # Do some validity checks (response is an 8-bit number [MSB]HGFEDCBA[LSB])
        val = int(resp)
        bit_str = bin(val).split("b")[1].zfill(8)

        bits = {"A": bit_str[-1], "B": bit_str[-2]}
        outp = int(bits[switch_id])

        print ("Switch", switch_id, "connected, Com =>", outp+1, "(VNA)" if outp+1 == 2 else "(USRP)")

        self.error = "Error, port did not change state"
        return  not(outp == out_port)

        

