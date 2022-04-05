#!/usr/bin/env/python
# start_gui.py

try:
    from tkinter import *
except ImportError:
    from TKinter import *

from laser_driver import *

## Create a driver instance
driver = LaserDriver()

## Define some colors
SLATEGREY = "#778899"
DANGERRED = "#ff0000"

## Geometry of the GUI
win_size_x  = 470
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
win.title('NEXUS Laser Contol')

## Set the size of the window
win.geometry( str(int(win_size_x)) + "x" + str(int(win_size_y)) )

## ========= BUTTONS ========= ##

## Details of the "configure serial" button
btn_cfg_serial=Button(win, text="Configure\nSerial Port", 
    width   = but_size_x,
    height  = but_size_y,
    state   = "normal", )
btn_cfg_serial.grid(row=0, column=0)

## Details of the "roll call" button
btn_roll_call=Button(win, text="Roll Call", 
    width   = but_size_x,
    height  = but_size_y,
    state   = "disabled", )
btn_roll_call.grid(row=0, column=1)

## Details of the "LED ON" button
btn_led_on=Button(win, text="LED On", 
    width   = int(but_size_x),
    height  = int(but_size_y/2),
    state   = "disabled", )
btn_led_on.grid(row=5, column=0)

## Details of the "LED OFF" button
btn_led_off=Button(win, text="LED Off", 
    width   = int(but_size_x),
    height  = int(but_size_y/2),
    state   = "disabled", )
btn_led_off.grid(row=5, column=1)

## Details of the "LED OFF" button
btn_disconnect=Button(win, text="Disconnect", 
    width   = but_size_x,
    height  = but_size_y,
    state   = "disabled", )
btn_disconnect.grid(row=1, column=0)

## Details of the "Update BF" button
btn_update_bf=Button(win, text="Update", 
    width   = int(but_size_x/2),
    height  = int(but_size_y/2),
    state   = "disabled", )
btn_update_bf.grid(row=1, column=3)

## Details of the "Update PW" button
btn_update_pw=Button(win, text="Update", 
    width   = int(but_size_x/2),
    height  = int(but_size_y/2),
    state   = "disabled", )
btn_update_pw.grid(row=2, column=3)

## Details of the "Update BF" button
btn_update_lr=Button(win, text="Update", 
    width   = int(but_size_x/2),
    height  = int(but_size_y/2),
    state   = "disabled", )
btn_update_lr.grid(row=3, column=3)

## ========= LABELS ========= ##
# Label(win, text='Roll Call: ').grid(row=0, column=1)
Label(win, text='Burst Freq. [Hz]:').grid(row=1, column=1)
Label(win, text='Pulse Width [us]:').grid(row=2, column=1)
Label(win, text='  Laser R. [int]:').grid(row=3, column=1)

l_en = Label(win, text='LASER OFF', font=('Times', 24))
l_en.grid(row=4, column=1)
l_en.config(bg=SLATEGREY)

## ========= TEXT BOXES ========= ##
## Create containers for the entry strings
id_str = StringVar() ; id_str.set("Identity")
pw_str = StringVar() ; pw_str.set("PulseWidth")
bf_str = StringVar() ; bf_str.set("BurstFreq")
lr_str = StringVar() ; lr_str.set("LaserR")

e_id = Entry(win, textvariable=id_str, state="disabled")
e_id.grid(row=0, column=2)

e_bf = Entry(win, textvariable=bf_str, state="disabled")
e_bf.grid(row=1, column=2)

e_pw = Entry(win, textvariable=pw_str, state="disabled")
e_pw.grid(row=2, column=2)

e_lr = Entry(win, textvariable=lr_str, state="disabled")
e_lr.grid(row=3, column=2)

## ========= CHECK BOXES ========= ##
laser_en = IntVar()
c_en = Checkbutton(win, text='Enable Laser',variable=laser_en, onvalue=1, offvalue=0, state="disabled")
c_en.grid(row=4, column=0)

def en_laser():
    if (laser_en.get()):
        l_en.config(bg=DANGERRED, text='LASER  ON')
    else:
        l_en.config(bg=SLATEGREY, text='LASER OFF')
    driver.enable_laser(laser_en.get())
c_en["command"] = en_laser

## ======================================= ##

def cfg_serial():
    btn_cfg_serial["state"] = "disabled"
    
    driver.configure_serial()
    
    btn_roll_call["state"]  = "normal"
    btn_disconnect["state"] = "normal"

    return
btn_cfg_serial["command"] = cfg_serial

def roll_call():
    # btn_roll_call["state"] = "disabled"

    id_str.set( driver.get_identity() )
    pw_str.set( driver.get_pw() )
    bf_str.set( driver.get_bf() )
    lr_str.set( driver.get_lr() )

    btn_led_on["state"]    = "normal"
    btn_led_off["state"]   = "normal"
    btn_update_pw["state"] = "normal"
    btn_update_bf["state"] = "normal"
    btn_update_lr["state"] = "normal"
    e_pw["state"] = "normal"
    e_bf["state"] = "normal"
    e_lr["state"] = "normal"
    c_en["state"] = "normal"

    return
btn_roll_call['command'] = roll_call

def led_on():
    btn_led_on["state"]  = "disabled"
    driver.led(True)
    btn_led_off["state"] = "normal"
    return
btn_led_on['command'] = led_on

def led_off():
    btn_led_off["state"] = "disabled"
    driver.led(False)
    btn_led_on["state"]  = "normal"
    return
btn_led_off['command'] = led_off

def disconnect():
    driver.close()
    return
btn_disconnect['command'] = disconnect

def update_pw():
    driver.set_pw( pw_str.get() )
    pw_str.set( driver.get_pw() )
    return
btn_update_pw['command'] = update_pw

def update_bf():
    driver.set_bf( bf_str.get() )
    bf_str.set( driver.get_bf() )
    return
btn_update_bf['command'] = update_bf

def update_lr():
    driver.set_lr( lr_str.get() )
    lr_str.set( driver.get_lr() )
    return
btn_update_lr['command'] = update_lr

## Execute
win.mainloop() #running the loop that works as a trigger
