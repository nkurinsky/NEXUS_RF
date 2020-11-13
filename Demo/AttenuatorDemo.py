from Attenuator import *

verbose=False
portname="/dev/cu.usbmodemR32809502262"

print("Testing USB Attenuator")
att = Attenuator(portName=portname,verbose=verbose)
print(att.getStatus())

print('Set Attenuation Low')
att.set(1,10.0)

print(att.getStatus())

print('Turn On')
att.setOn()

print(att.getStatus())

print('Turn Off')
att.setOff()

print(att.getStatus())
