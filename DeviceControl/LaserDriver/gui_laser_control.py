#! /usr/bin/env python
# Arduino_Driver.py

import sys
import glob
import serial
import time
import json
from os import path
try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

import pyodbedit

odbbase='/Custom/NexusLaserDriverpyGUI/Settings/' #Base path to HV settings in odb
doODBwrite=True

ArduinoSerial = serial.Serial() # open serial port
#ArduinoSerial = serial.Serial('/dev/ttyUSB0', 115200, timeout=.1) # open serial port
time.sleep(0.5)
LARGE_FONT= ("Verdana", 12)

def sendCommandToArduino(cmd):
    #print ('Sending command to Arduino: %s'%cmd)
    ArduinoSerial.write(str.encode(cmd)) # send command
    #line=ArduinoSerial.readline()
    #print(line)

def bool2yn(val):
    if val:
        return 'y'
    else:
        return 'n'
def yn2bool(val):
    if val.lower()=='y':
        return True
    else:
        return False

class LaserControl(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)
        self.winfo_toplevel().title("Laser Controller")
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo):

            frame = F(parent=container, controller=self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

    def updateSettingsFromODB(self):
        if not self.frames[PageTwo].LocalControl.get():
            #print('updateSettingsFromODB')
            self.frames[PageTwo].setAllFromODB()
        self.after(1000, self.updateSettingsFromODB)


class FreqSlideBar(tk.Frame):
    def __init__(self, parent, controller,label=''):
        tk.Frame.__init__(self,parent)
        if label != '':
            self.title = tk.Label(self, text=label)
            self.title.pack()
        self.sliderValue = tk.DoubleVar()
        self.w1 = tk.Scale(self, from_=0.001, to=99.990, orient=tk.VERTICAL, resolution=0.001, showvalue=0, variable=self.sliderValue)
        self.w1.set(1.0)
        self.w1.bind("<ButtonRelease-1>",lambda event: self.update_slider())
        self.w1.pack()
        self.e1 = tk.Entry(self, width=5)
        self.e1.insert(0,self.sliderValue.get())
        self.e1.pack()
        self.e1.bind("<Return>",lambda event : self.setw1(self.e1.get()))
        self.e1.bind("<KP_Enter>",lambda event : self.setw1(self.e1.get()))
        self.e1.bind("<FocusOut>",lambda event : self.setw1(self.e1.get()))
    def update_slider(self):
        if self.master.LocalControl.get():
            self.e1.delete(0,tk.END)
            self.e1.insert(0,self.sliderValue.get())
            self.sendCommand()

            ### Update to ODB
            self.writeToODB()

    def setw1(self ,w1value):
        if self.master.LocalControl.get():
            self.w1.set(w1value)
            self.sendCommand()

            ### Update to ODB
            self.writeToODB()

    def sete1value(self,e1value):
        if e1value != self.sliderValue:
            self.sliderValue.set(float(e1value))
            self.sendCommand()
        self.e1.delete(0,tk.END)
        self.e1.insert(0,str(e1value))
    def sendCommand(self):
        cmd = "F%5.2f\n"%(self.sliderValue.get())
        #print ("Sending command for SQUID card, CH %d: %s"%(self.CHNum,cmd))
        sendCommandToArduino(cmd)

    ### When local control, update configuration in ODB.
    def writeToODB(self):
        #PulseFreq
        writeOK=pyodbedit.write(f"{odbbase}PulseFreq(MHz)",f'{self.w1.get():.4f}')
        if writeOK!='':
            print('Error in writeToODB. writeOK=',writeOK)



class BurstSlideBar(tk.Frame):
    def __init__(self, parent, controller,label=''):
        tk.Frame.__init__(self,parent)
        if label != '':
            self.title = tk.Label(self, text=label)
            self.title.pack()
        self.sliderValue = tk.DoubleVar()
        self.w1 = tk.Scale(self, from_=0.01, to=999.9, orient=tk.VERTICAL, resolution=0.001, showvalue=0, variable=self.sliderValue)
        self.w1.set(10.0)
        self.w1.bind("<ButtonRelease-1>",lambda event: self.update_slider())
        self.w1.pack()
        self.e1 = tk.Entry(self, width=5)
        self.e1.insert(0,self.sliderValue.get())
        self.e1.pack()
        self.e1.bind("<Return>",lambda event : self.setw1(self.e1.get()))
        self.e1.bind("<KP_Enter>",lambda event : self.setw1(self.e1.get()))
        self.e1.bind("<FocusOut>",lambda event : self.setw1(self.e1.get()))
    def update_slider(self):
        if self.master.LocalControl.get():
            self.e1.delete(0,tk.END)
            self.e1.insert(0,self.sliderValue.get())
            self.sendCommand()

            ### Update to ODB
            self.writeToODB()
    def setw1(self ,w1value):
        if self.master.LocalControl.get():
            self.w1.set(w1value)
            self.sendCommand()
    def sete1value(self,e1value):
        if e1value != self.sliderValue:
            self.sliderValue.set(float(e1value))
            self.sendCommand()
        self.e1.delete(0,tk.END)
        self.e1.insert(0,e1value)
    def sendCommand(self):
        cmd = "B%5.2f\n"%(self.sliderValue.get())
        #print ("Sending command for SQUID card, CH %d: %s"%(self.CHNum,cmd))
        sendCommandToArduino(cmd)

    ### Update to ODB
    def writeToODB(self):
        #BurstFreq
        writeOK=pyodbedit.write(f"{odbbase}BurstFreq(Hz)",f'{self.w1.get():.4f}')
        if writeOK!='':
            print('Error in writeToODB. writeOK=',writeOK)


class RSlideBar(tk.Frame):
    def __init__(self, parent, controller,label=''):
        tk.Frame.__init__(self,parent)
        if label != '':
            self.title = tk.Label(self, text=label)
            self.title.pack()
        self.sliderValue = tk.IntVar()
        self.w1 = tk.Scale(self, from_=0, to=127, orient=tk.VERTICAL, resolution=1, showvalue=0, variable=self.sliderValue)
        self.w1.set(127.0)
        self.w1.bind("<ButtonRelease-1>",lambda event: self.update_slider())
        self.w1.pack()
        self.e1 = tk.Entry(self, width=5)
        self.e1.insert(0,self.sliderValue.get())
        self.e1.pack()
        self.e1.bind("<Return>",lambda event : self.setw1(self.e1.get()))
        self.e1.bind("<KP_Enter>",lambda event : self.setw1(self.e1.get()))
        self.e1.bind("<FocusOut>",lambda event : self.setw1(self.e1.get()))
    def update_slider(self):
        if self.master.LocalControl.get():
            self.e1.delete(0,tk.END)
            self.e1.insert(0,self.sliderValue.get())
            self.sendCommand()

            ### Update to ODB
            self.writeToODB()

    def setw1(self ,w1value):
        if self.master.LocalControl.get():
            self.w1.set(w1value)
            self.sendCommand()

            ### Update to ODB
            self.writeToODB()

    def sete1value(self,e1value):
        if e1value != self.sliderValue:
            self.sliderValue.set(float(e1value))
            self.sendCommand()
        self.e1.delete(0,tk.END)
        self.e1.insert(0,e1value)
    def sendCommand(self):
        cmd = "R%d\n"%(self.sliderValue.get())
        #print ("Sending command for SQUID card, CH %d: %s"%(self.CHNum,cmd))
        sendCommandToArduino(cmd)

    ### Update to ODB
    def writeToODB(self):
        #PowerControl
        writeOK=pyodbedit.write(f"{odbbase}PowerControl",f'{self.w1.get():.4f}')
        if writeOK!='':
            print('Error in writeToODB. writeOK=',writeOK)



class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Configuration", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        #-- Looking for serial ports available
        self.serial_ports_available = self.serial_ports()
        self.var_serial_port = tk.StringVar()
        #  print(self.serial_ports_available, len(self.serial_ports_available))
        self.var_serial_port.set(self.serial_ports_available[0])

        #-- Choose serial ports from available list
        self.port_optionMenu = tk.OptionMenu(self,self.var_serial_port, *self.serial_ports_available)
        self.port_optionMenu.pack()

        #-- Configure serial ports, currently defaulting to 115200 baudrate, 0.1s timeout
        btn_configSerial = tk.Button(self, text='Configure Serial Port', bg = 'light green',  command=self.configSerial) # activate Arduino pin 13
        btn_configSerial.pack()


        #-- Roll call, to confirm the identity of the minion
        self.var_IDN=tk.StringVar()
        self.var_IDN.set("Minion absent")

        btn_rollcall = tk.Button(self, text='Roll call', bg = 'light green', command=self.roll_call) # activate Arduino pin 13
        btn_rollcall.pack()
        lbl_rollcall = tk.Label (self, textvariable = self.var_IDN, bg = 'light blue', fg = 'black', highlightbackground= 'black',\
                borderwidth = 3, relief=tk.RAISED, font=("Helvetica", 24), width=16, justify=tk.CENTER)
        lbl_rollcall.pack(pady=5)



        button = tk.Button(self, text="Minion Health Check",
                command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = tk.Button(self, text="Minion Performance Stage",
                command=lambda: controller.show_frame(PageTwo), )
        button2.pack()
        btn_exit = tk.Button(self, text='Exit',  bg = 'pink',   \
                borderwidth = 0,command=self.arduino_Exit) # close serial port and quit program
        btn_exit.pack()

    def configSerial(self):
        ArduinoSerial.port = self.var_serial_port.get()
        ArduinoSerial.baudrate = 115200
        ArduinoSerial.timeout = 0.5
        try:
            ArduinoSerial.open()
        except Exception as e:
            print ("error open serial port: "+str(e))

    def roll_call(self):
        #ArduinoSerial.reset_input_buffer()
        ArduinoSerial.write(str.encode('*IDN?\n')) # mimicing GPIB protocal
        try:
            self.lineinput=ArduinoSerial.readline()
        except serial.SerialException as e:
            self.var_IDN.set("Minion not recognized")
        except Exception as e:
            print ("error doing roll-call: "+str(e))
        else:
            if self.lineinput == "":
                self.var_IDN.set("Minion did not respond to roll call")
            else:
                self.var_IDN.set(self.lineinput.decode('utf-8').strip())

    def serial_ports(self):
        """ Lists serial port names
        :raises EnvironmentError:
        On unsupported or unknown platforms
        :returns:
        A list of the serial ports available on the system
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')
        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

    def arduino_Exit(self):
        try:
            ArduinoSerial.write('0\n') # set Arduino output pin 13 low and quit
            ArduinoSerial.close() # close serial port
        except:
            pass
        quit()

class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, )
        label = tk.Label(self, text="Minion Health Check", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.var1 = tk.StringVar()
        self.var1.set('OFF')


        btn1 = tk.Button(self, text='Led ON', bg = 'light green', highlightbackground= 'black',   \
                borderwidth = 3, activebackground = 'light gray', relief=tk.RAISED, command=self.led_on) # activate Arduino pin 13
        btn1.pack()

        btn2 = tk.Button(self, text='Led OFF', bg = 'red2', fg = 'white', highlightbackground= 'black',   \
                borderwidth = 3, activebackground = 'light gray', relief=tk.RAISED, command=self.led_off) # deactivate Arduino pin 13
        btn2.pack()
        lbl1 = tk.Label (self, textvariable = self.var1, bg = 'light blue', fg = 'black', highlightbackground= 'black',\
                borderwidth = 3, relief=tk.RAISED)
        lbl1.pack()


        button1 = tk.Button(self, text="Back to Configuration",
                command=lambda: controller.show_frame(StartPage), )
        button1.pack()

        button2 = tk.Button(self, text="Minion Performance Stage",
                command=lambda: controller.show_frame(PageTwo), )
        button2.pack()
        btn_exit = tk.Button(self, text='Exit',  bg = 'pink',   \
                borderwidth = 0,command=self.arduino_Exit) # close serial port and quit program
        btn_exit.pack()


    def led_on(self):
        #ArduinoSerial.reset_input_buffer()
        ArduinoSerial.write(str.encode('*LED1\n')) # set Arduino output pin 13 high
        line=ArduinoSerial.readline()
        #print('receiving from serial:'+line)
        self.var1.set(line.decode('utf-8').strip())# get Arduino output pin 13 status
        if self.var1.get() == 'LED1':
            self.var1.set('Pin 13 is ON')

    def led_off(self):
        ArduinoSerial.write(str.encode('*LED0\n')) # set Arduino output pin 13 low
        self.var1.set(ArduinoSerial.readline().decode('utf-8').strip()) # get Arduino output pin 13 status
        if self.var1.get() == 'LED0':
            self.var1.set('Pin 13 is OFF')

    def arduino_Exit(self):
        try:
            ArduinoSerial.write(str.encode('0\n')) # set Arduino output pin 13 low and quit
            ArduinoSerial.close() # close serial port
        except:
            pass
        quit()





class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, )
        label = tk.Label(self, text="Let's put the minions to work.", font=LARGE_FONT)
        label.grid(row=0, columnspan=3)
        self.OnOff = tk.BooleanVar()
        self.OnOff.set(False)
        self.FreqBar = FreqSlideBar(parent=self, controller=self, label="Pulse Freq\n(MHz)")
        self.FreqBar.grid(row=1, column=0)
        self.BurstBar = BurstSlideBar(parent=self, controller=self, label="Burst Freq\n(Hz)")
        self.BurstBar.grid(row=1, column=1)
        self.RBar = RSlideBar(parent=self, controller=self, label="Power Control\nP ~ 1/R")
        self.RBar.grid(row=1, column=2)


        # self.var_ACK = tk.StringVar()
        # self.var_ACK.set('Idle')
        # lbl_ACK = tk.Label (self, textvariable = self.var_ACK, bg = 'light blue', fg = 'black', highlightbackground= 'black',\
                #         borderwidth = 3, relief=tk.RAISED)
        # lbl_ACK.grid(row=2,columnspan=2)
        self.laser_status = tk.StringVar()
        self.laser_status.set('Laser is OFF')


        self.btn_on = tk.Button(self, text='Laser ON', bg = 'light green', highlightbackground= 'black',   \
                borderwidth = 0, activebackground = 'light gray', relief=tk.RAISED, command=self.laser_on) # activate Arduino pin 13
        self.btn_on.grid( row=2, column=1)

        self.btn_off = tk.Button(self, text='Laser OFF', bg = 'DarkSeaGreen4', fg = 'white', highlightbackground= 'black',   \
                borderwidth = 0, activebackground = 'light gray', relief=tk.RAISED, command=self.laser_off) # deactivate Arduino pin 13
        self.btn_off.grid(row=3, column=1)
        self.lbl_laser = tk.Label (self, textvariable = self.laser_status, bg='cornsilk2', fg='black', highlightbackground= 'black',\
                borderwidth = 3, font=("Helvetica", 30), width=15, justify=tk.CENTER)
        self.lbl_laser.grid(row=4, columnspan=3, pady=10)


        button1 = tk.Button(self, text="Back to Configuration",
                command=lambda: controller.show_frame(StartPage))
        button1.grid(row=7, columnspan=3)

        button2 = tk.Button(self, text="Minion Health Check",
                command=lambda: controller.show_frame(PageOne))
        button2.grid(row=8, columnspan=3)
        btn_exit = tk.Button(self, text='Exit',  bg = 'pink',   \
                borderwidth = 0,command=self.arduino_Exit) # close serial port and quit program
        btn_exit.grid(row=9, columnspan=3)

	## Local Control or Remote Control
        self.LocalControl=tk.BooleanVar()
        self.LocalControl.set(True)
        loc_button = tk.Radiobutton(self, text="GUI Control", variable=self.LocalControl, value=True, command = self.setLocalControl)
        rem_button = tk.Radiobutton(self, text="ODB Control", variable=self.LocalControl, value=False, command = self.setODBControl)
        loc_button.grid(row=5,columnspan=3)
        rem_button.grid(row=6,columnspan=3)

    def laser_on(self):
        #ArduinoSerial.reset_input_buffer()
        ArduinoSerial.write(str.encode('ON\n')) # set Arduino output pin 13 high
        #print('receiving from serial:'+line)
        self.laser_status.set('Laser is ON')
        self.lbl_laser.config(bg='OrangeRed2', fg='white')
        self.OnOff.set(True)
        
        ### Update to ODB
        if self.LocalControl.get():
            #On/Off
            writeOK=pyodbedit.write(f"{odbbase}Power",bool2yn(self.OnOff.get()))
            if writeOK!='':
                print('Error in writeToODB. writeOK=',writeOK)


    def laser_off(self):
        ArduinoSerial.write(str.encode('OFF\n')) # set Arduino output pin 13 low
        self.lbl_laser.config(bg='cornsilk2', fg='black')
        self.laser_status.set('Laser is OFF')
        self.OnOff.set(False)

        ### Update to ODB
        if self.LocalControl.get():
            #On/Off
            writeOK=pyodbedit.write(f"{odbbase}Power",bool2yn(self.OnOff.get()))
            if writeOK!='':
                print('Error in writeToODB. writeOK=',writeOK)


    def writeToODB(self):
        #On/Off
        writeOK=pyodbedit.write(f"{odbbase}Power",bool2yn(self.OnOff.get()))
        if writeOK!='':
            print('Error in writeToODB. writeOK=',writeOK)
        #PulseFreq
        writeOK=pyodbedit.write(f"{odbbase}PulseFreq(MHz)",f'{self.FreqBar.w1.get():.4f}')
        if writeOK!='':
            print('Error in writeToODB. writeOK=',writeOK)
        #BurstFreq
        writeOK=pyodbedit.write(f"{odbbase}BurstFreq(Hz)",f'{self.BurstBar.w1.get():.4f}')
        if writeOK!='':
            print('Error in writeToODB. writeOK=',writeOK)
        #PowerControl
        writeOK=pyodbedit.write(f"{odbbase}PowerControl",f'{self.RBar.w1.get():.4f}')
        if writeOK!='':
            print('Error in writeToODB. writeOK=',writeOK)

    def setAllFromODB(self):
        #On/Off
        OnOff = pyodbedit.read(f"{odbbase}Power")
        OnOff = yn2bool(OnOff)
        if OnOff:
            self.laser_on()
        else:
            self.laser_off()
        #PulseFreq
        PulseFreq = pyodbedit.read(f"{odbbase}PulseFreq(MHz)")
        self.FreqBar.sete1value(PulseFreq)
        #BurstFreq
        BurstFreq = pyodbedit.read(f"{odbbase}BurstFreq(Hz)")
        self.BurstBar.sete1value(BurstFreq)
        #PowerControl
        PowerControl = pyodbedit.read(f"{odbbase}PowerControl")
        self.RBar.sete1value(PowerControl)


    def setLocalControl(self):
        print("LocalControl:",self.LocalControl.get())
    def setODBControl(self):
        #Transition to giving ODB control
        print("LocalControl:",self.LocalControl.get())
        self.writeToODB()

    def arduino_Exit(self):
        try:
            laser_off() # turn off laser and quit
            ArduinoSerial.close() # close serial port
        except:
            pass
        quit()


app = LaserControl()
app.after(1000, app.updateSettingsFromODB)
app.mainloop()
