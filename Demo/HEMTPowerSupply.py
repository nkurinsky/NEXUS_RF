import sys
sys.path.append("../Devices")

from E36300 import *

hemts = [E36300("192.168.0.40"),E36300("192.168.0.41")]

for i in range(1,3):
    print("HEMT "+str(i))
    hemt = hemts[i-1]
    print(hemt.getID())
    print(hemt.getVoltage())
    print(hemt.getCurrent())
    print(hemt.getStatus())
    print("")
