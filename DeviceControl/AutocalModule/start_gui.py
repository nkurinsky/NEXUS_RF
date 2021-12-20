from comms_driver import *
## Import the GUI modules
try:
    from tkinter import *
    import tkinter.messagebox
except ImportError:
    from TKinter import *
    import TKinter.messagebox

## Geometry of the GUI
win_size_x  = 500
win_size_y  = 180

but_size_x  =  12
but_size_y  =   2

but1_pos_x  =   0
but1_pos_y  =  70

but2_pos_x  =   0
but2_pos_y  = 100

col2_pos_x  = 125

lbl_pos_y   =  10
txt_pos_y   =  30 
txt_size_x  = win_size_x - col2_pos_x - 10 -314
txt_size_y  = win_size_y - txt_pos_y  - 10 -129

## Create the main window and storing the window object in 'win'
win=Tk() 

## Set the title of the window
win.title('NEXUS Autocalibration Module Contol')

## Set the size of the window
win.geometry( str(int(win_size_x)) + "x" + str(int(win_size_y)) )

## ============= INFO FIELDS ============= ##
## Create labels and fields for text display

Label(win, text='System response:').place(x=col2_pos_x,y=lbl_pos_y)

## Create containers for the entry strings
t1 = Text(win,
    width   = txt_size_x,
    height  = txt_size_y,)
t1.place(x=col2_pos_x,y=txt_pos_y)
## ======================================= ##



## ============= MODE BUTTONS ============ ##
## Create button objects to switch the mode

b_thru = Button(win, text="Through", 
    width   = but_size_x,
    height  = but_size_y,)
b_attn = Button(win, text="Attenuator", 
    width   = but_size_x,
    height  = but_size_y,)
b_shrt = Button(win, text="Short", 
    width   = but_size_x,
    height  = but_size_y,)
b_open = Button(win, text="Open", 
    width   = but_size_x,
    height  = but_size_y,)
b_load = Button(win, text="Load", 
    width   = but_size_x,
    height  = but_size_y,)

b_thru.grid(row=0, column=0)
b_attn.grid(row=1, column=0)
b_shrt.grid(row=2, column=0)
b_open.grid(row=3, column=0)
b_load.grid(row=4, column=0)
## ======================================= ##



## =========== BUTTON FUNCTIONS ========== ##
## Create the functions to execute on a button press

def b_thru_press():
    ans = Send("SYST:COMM:ECAL:THRU", arg_str="1,2")
    t1.insert(END, ans)
    return 0
b_thru["command"] = b_thru_press

def b_attn_press():
    ans = Send("SYST:COMM:ECAL:CHEC")
    t1.insert(END, ans)
    return 0
b_attn["command"] = b_attn_press

def b_shrt_press():
    ans = Send("SYST:COMM:ECAL:IMP", arg_str="1,SHORT")
    t1.insert(END, ans)
    ans = Send("SYST:COMM:ECAL:IMP", arg_str="2,SHORT")
    t1.insert(END, ans)
    return 0
b_shrt["command"] = b_shrt_press

def b_open_press():
    ans = Send("SYST:COMM:ECAL:IMP", arg_str="1,OPEN")
    t1.insert(END, ans)
    ans = Send("SYST:COMM:ECAL:IMP", arg_str="2,OPEN")
    t1.insert(END, ans)
    return 0
b_open["command"] = b_open_press

def b_load_press():
    ans = Send("SYST:COMM:ECAL:IMP", arg_str="1,LOAD")
    t1.insert(END, ans)
    ans = Send("SYST:COMM:ECAL:IMP", arg_str="2,LOAD")
    t1.insert(END, ans)
    return 0
b_load["command"] = b_load_press

## ======================================= ##

## Execute
win.mainloop() #running the loop that works as a trigger
