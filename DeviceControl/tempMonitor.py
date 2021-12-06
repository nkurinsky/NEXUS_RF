import sys
sys.path.append("../Devices/")

from tkinter import *
from tkinter import messagebox
from ADRfunctions import *

debug=False

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
    ttVar.set(formatTemp(fridge.getTempSP()))

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
ttLab = Label(root, text = 'Target Temp:').grid(row = 1, column = 0)
sLab = Label(root, text = 'Set Point:').grid(row = 2, column = 0)
rLab = Label(root, text = 'Ramp 1:').grid(row = 3, column = 0)
rLab2 = Label(root, text = 'Ramp 2:').grid(row = 4, column = 0)

tLab2 = Label(root, text = "New Temp Target:").grid(row=6,column=0)
tLab2 = Label(root, text = "New Set Point:").grid(row=7,column=0)

tVar = StringVar()
ttVar = StringVar()
sVar = StringVar()
sVarSet = StringVar()
rVar = StringVar()
rVar2 = StringVar()
tDisplay = Label(root, textvariable = tVar).grid(row = 0, column = 1)
ttDisplay = Label(root, textvariable = ttVar).grid(row=1, column = 1)
sDisplay = Label(root, textvariable = sVar).grid(row = 2, column = 1)
rDisplay = Label(root, textvariable = rVar).grid(row = 3, column = 1)
rDisplay2 = Label(root, textvariable = rVar2).grid(row = 4, column = 1)

sPt = Entry(root)
sPt.grid(row=7,column=1)

stPt = Entry(root)
stPt.grid(row=6,column=1)

def tempSetError():
    messagebox.showerror("Invalid Entry","Temperature must be between base temperature and 4K")

def tempSetWarning():
    return messagebox.askyesno("Low Set Point Warning","The requested set point is close to the expected base temperature of the ADR, and may drive the PID into a bad state. Are you sure you want to continue?")

def setSP(tempMode=False):
    if(tempMode):
        newSP = float(stPt.get())
        tempMin = 0.085
    else:
        newSP = float(sPt.get())
        tempMin = 0.06

    if(debug):
        print(newSP)

    if(newSP < 0):
        if(debug):
            print("Invalid Temp (Too Low)")
        tempSetError()
        return
    if(newSP > 4):
        if(debug):
            print("Invalid Temp (Too High)")
        tempSetError()
        return

    if(newSP < tempMin):
        if(debug):
            print("Warning: Temp Close to Minimum Value")
        cont = tempSetWarning()
        if(not cont):
            return

    if(tempMode):
        fridge.setTempSP(newSP)
    else:
        fridge.setSP(newSP)
    getSP()
    getRamp()

def setTempSP():
    setSP(tempMode=True)

stButton = Button(root, text="Set", command=setTempSP).grid(row=6,column=2)
sButton = Button(root, text="Set", command=setSP).grid(row=7,column=2)
fButton = Button(root, text="Refresh", command=updateAll).grid(row=5,column=0)
rButton = Button(root, text="Ramp Off", command=rampOff).grid(row=5,column=1)

updateTemp()
updateValues()
root.mainloop()
