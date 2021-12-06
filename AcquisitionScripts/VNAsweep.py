import sys

sys.path.append("../Devices/")
from VNAfunctions import *

vna=VNA()

vna.setPower(-45)

f,s=vna.takeSweep(4241e6,4246e6,50000, 8)

print(len(s))
print(s[49999])
