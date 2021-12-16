## Point to the backend function scripts
sys.path.insert(1, "/home/nexus-admin/NEXUS_RF/DeviceControl")

from VNAfunctions import *  #using the VNA to do a power sweep

## Create an instance of the VNA class
v = VNA()

## Send command wrapper
def Send(msg, arg_str=None):

    ## Create a command with no arguments
    if (arg_str==None):
        msg_str = str(msg)+"\n"

    ## Create a command with arguments
    else:
        msg_str = str(msg)+" "+str(arg_str)+"\n"
    
    ## Send the command    
    v._sendCmd(msg_str)

    ## Query the device for the state
    r = v._getData(str(msg)+"?\n")

    ## Return the message, query, and answer as one string
    return msg_str+str(msg)+"?\n"+r+"\n"