from comms_driver import *
try:
    from tkinter import *
    import tkinter.messagebox
except ImportError:
    from TKinter import *
    import TKinter.messagebox

## Instantiate the RF Switch
ip   = "192.168.0.43"
rfsw = IPInstrument(ip)

## Geometry of the GUI
win_size_x  = 500
win_size_y  = 200

but_size_x  =  12
but_size_y  =   2

but1_pos_x  =   0
but1_pos_y  =  70

but2_pos_x  =   0
but2_pos_y  = 100

## Create the main window and storing the window object in 'win'
win=Tk() 

## Set the title of the window
win.title('NEXUS RF Switch Contol')

## Set the size of the window
win.geometry( str(int(win_size_x)) + "x" + str(int(win_size_y)) )


## ========= PROBE FUNCTIONALITY ========= ##
## Create containers for the entry strings
sn_str = StringVar()
mn_str = StringVar()

## Add the labels and entries that get populated by probe
Label(win, text='RF Switch Model: ').grid(row=0, column=1)
Label(win, text='RF Switch Serial:').grid(row=1, column=1)
e1 = Entry(win, textvariable=mn_str, state="disabled")
e2 = Entry(win, textvariable=sn_str, state="disabled")
e1.grid(row=0, column=2)
e2.grid(row=1, column=2)

## Details of the "Probe" button
btn_probe=Button(win, text="Probe", 
    width   = but_size_x,
    height  = but_size_y,)
btn_probe.grid(row=0, column=0)
## ======================================= ##

## Create containers for the entry strings
cnx_str = StringVar()
cnx_str.set("None")

## Add the label and entry that get populated by other buttons
Label(win, text='Connected to:').place(x=110,y=70)
e3 = Entry(win, textvariable=cnx_str, state="disabled")
e3.place(x=235,y=70)

## Details of the button
btn_vna=Button(win, text="Connect VNA", 
    width   = but_size_x,
    height  = but_size_y,
    state   = "disabled")
btn_vna.place(x=but1_pos_x,y=but1_pos_y)

## Details of the button
btn_usrp=Button(win, text="Connect USRP", 
    width   = but_size_x,
    height  = but_size_y,
    state   = "disabled")
btn_usrp.place(x=but2_pos_x,y=but2_pos_y)

## Probe Button and function
def probe():
    mn, sn = rfsw.TestConnection()
    mn_str.set(mn)#"ModelNumber")
    sn_str.set(sn)#"SerialNumber")
    btn_vna["state"]  = "normal"
    btn_usrp["state"] = "normal"
btn_probe["command"] = probe

def func_vna():
    btn_usrp["state"] = "normal"
    btn_vna["state"]  = "disabled"
    cnx_str.set("VNA")

    if(rfsw.SetSwitchState("A", 2)):
        tkinter.messagebox.showinfo("Error",rfsw.error)
    if(rfsw.SetSwitchState("B", 2)):
        tkinter.messagebox.showinfo("Error",rfsw.error)
btn_vna['command'] = func_vna

def func_usrp():
    btn_vna["state"]  = "normal"
    btn_usrp["state"] = "disabled"
    cnx_str.set("USRP")

    if(rfsw.SetSwitchState("A", 1)):
        tkinter.messagebox.showinfo("Error",rfsw.error)
    if(rfsw.SetSwitchState("B", 1)):
        tkinter.messagebox.showinfo("Error",rfsw.error)
btn_usrp['command'] = func_usrp


## Execute
win.mainloop() #running the loop that works as a trigger
