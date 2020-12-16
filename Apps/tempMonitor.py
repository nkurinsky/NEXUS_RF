import sys
sys.path.append("../Devices/")

from tkinter import *
from ADRfunctions import *

fridge = ADR()

def formatTemp(temp):
    if(temp < 1):
        return str(temp*1e3)[:5]+' mK'
    else:
        return str(temp)[:5]+' K'

def getTemp():
    return formatTemp(fridge.getTemp())

def getSP():
    sVar.set(formatTemp(fridge.getSP()))

def getRamp():
    ramp = fridge.getRamp()
    if(ramp[0]):
        ramp1 = "True"
    else:
        ramp1 = "False"

    if(ramp[1]):
        ramp2 = "True"
    else:
        ramp2 = "False"

    rVar.set(ramp1)
    rVar2.set(ramp2)

def updateTemp():
    tVar.set(getTemp())
    root.after(10000, updateTemp)

def updateValues():
    getSP()
    getRamp()
    root.after(60000, updateValues)

def updateAll():
    tVar.set(getTemp())
    getSP()
    getRamp()

def rampOff():
    fridge.rampOff()
    getRamp()

root = Tk()
root.title('ADR Status')

tLab = Label(root, text = 'ADR Temp:').grid(row = 0, column = 0)
sLab = Label(root, text = 'Set Point:').grid(row = 1, column = 0)
rLab = Label(root, text = 'Ramp 1:').grid(row = 2, column = 0)
rLab2 = Label(root, text = 'Ramp 2:').grid(row = 3, column = 0)

tLab2 = Label(root, text = "New Set Point:").grid(row=5,column=0)

tVar = StringVar()
sVar = StringVar()
sVarSet = StringVar()
rVar = StringVar()
rVar2 = StringVar()
tDisplay = Label(root, textvariable = tVar).grid(row = 0, column = 1)
sDisplay = Label(root, textvariable = sVar).grid(row = 1, column = 1)
rDisplay = Label(root, textvariable = rVar).grid(row = 2, column = 1)
rDisplay2 = Label(root, textvariable = rVar2).grid(row = 3, column = 1)

sPt = Entry(root)
sPt.grid(row=5,column=1)

def setSP():
    newSP = float(sPt.get())
    print(newSP)

    if(newSP < 0):
        print("Invalid Temp (Too Low)")
        return
    if(newSP > 4):
        print("Invalid Temp (Too High)")
        return

    if(newSP < 0.06):
        print("Warning: Temp Close to Minimum Value")
        
    fridge.setSP(newSP)
    getSP()
    getRamp()

sButton = Button(root, text="Set", command=setSP).grid(row=5,column=2)
fButton = Button(root, text="Refresh", command=updateAll).grid(row=4,column=0)
rButton = Button(root, text="Ramp Off", command=rampOff).grid(row=4,column=1)

updateTemp()
updateValues()
root.mainloop()
