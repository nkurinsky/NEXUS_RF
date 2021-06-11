import sys
from time import sleep
sys.path.append("../Devices")

from E36300 import *

hemts = [E36300("192.168.0.40"),E36300("192.168.0.41")]

for i in range(1,3):
    print("HEMT "+str(i))
    hemt = hemts[i-1]
    print("Status")
    print(hemt.getStatus())
    print("Disabling")
    hemt.disable([2,3])
    sleep(1)
    print("New Status")
    print(hemt.getVoltage())
    print(hemt.getCurrent())
    print(hemt.getStatus())
    print("")
