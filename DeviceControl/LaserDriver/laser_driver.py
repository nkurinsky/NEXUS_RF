#!/usr/bin/env/python
# laser_driver.py

import sys
import glob
import serial
import time

def pw_to_freq(pw_us):
	## Convert a "pulse width" in microseconds
	## to a "frequency" in MHz, assuming 50% duty cycle
	return 1./(2.*pw_us)

def freq_to_pw(fq_MHz):
	## Convert a "frequency" in MHz
	## to a "pulse width" in microseconds, assuming 50% duty cycle
	return 1./(2.*fq_Hz)

def bool2yn(val):
    return 'y' if val else 'n'

def yn2bool(val):
    return val.lower()=='y'

def serial_ports():
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

class LaserDriver():

	ArduinoSerial = None  # serial connection object
	serial_port   = None  # string pointing to serial port address

	def __init__(self):
		## Instantiate the serial connection object
		self.ArduinoSerial = serial.Serial() # ('/dev/ttyUSB0', 115200, timeout=.1)
		time.sleep(0.5)

		#-- Looking for serial ports available
		serial_ports_available = serial_ports()
		try:
			self.serial_port = serial_ports_available[0]
		except IndexError:
			print("No serial ports available, exiting")
			quit()
		return

	def _sendCmd(self, cmd):
		#print('Sending command to Arduino: %s'%cmd)
		self.ArduinoSerial.write(str.encode(cmd+'\n'))
		return

	def _getReply(self):
		line = "No response."
		try:
			line = self.ArduinoSerial.readline()
		except Exception as e:
		    print ("Error in reply: "+str(e))
		#print(line)
		return line.decode('utf-8').strip()

	def configure_serial(self):
		self.ArduinoSerial.port     = self.serial_port
		self.ArduinoSerial.baudrate = 115200
		self.ArduinoSerial.timeout  = 0.5
		try:
			self.ArduinoSerial.open()
		except Exception as e:
			print("Error opening serial port: "+str(e))
		return
	
	def get_identity(self):
		self._sendCmd("*IDN?")
		time.sleep(0.05)
		return self._getReply()

	def led(self, state=False):
		cmd = "*LED" + ("1" if state else "0")
		self._sendCmd(cmd)
		time.sleep(0.05)
		print(self._getReply())
		return

	def set_pw(self, pw_us="10.00"):
		try:
			pw = float(pw_us)
		except Exception as e:
			print("Error converting",pw_us,"to float -- ",str(e))
			return

		clock_f = pw_to_freq(pw)

		if (clock_f < 0.001):
			clock_f = 0.001

		if (clock_f > 99.99):
			clock_f = 99.99

		try:
			cmd = "F%5.2f"%clock_f
		except Exception as e:
			print("Error converting",clock_f,"to string -- ",str(e))
			return

		self._sendCmd(cmd)
		time.sleep(0.05)
		print(self._getReply())
		return

	def get_pw(self):
		self._sendCmd("F?")
		time.sleep(0.05)
		pw_MHz = float(self._getReply())
		return string(freq_to_pw(pw_MHz))

	def set_bf(self, bf_Hz="250.0"):
		try:
			bf = float(bf_Hz)
		except Exception as e:
			print("Error converting",bf_Hz,"to float -- ",str(e))
			return

		if (bf < 5.0):
			bf = 5.0

		if (bf > 999.9):
			bf = 999.9

		try:
			cmd = "B%5.2f"%bf
		except Exception as e:
			print("Error converting",bf,"to string -- ",str(e))
			return

		self._sendCmd(cmd)
		time.sleep(0.05)
		print(self._getReply())
		return

	def get_bf(self):
		self._sendCmd("B?")
		time.sleep(0.05)
		return self._getReply()

	def set_lr(self, lr_R="127"):
		try:
			R = int(lr_R)
		except Exception as e:
			print("Error converting",lr_R,"to float -- ",str(e))
			return

		if (R < 1):
			R = 1

		if (R > 127):
			R = 127

		try:
			cmd = "R%d"%R
		except Exception as e:
			print("Error converting",R,"to string -- ",str(e))
			return

		self._sendCmd(cmd)
		time.sleep(0.05)
		print(self._getReply())
		return

	def get_lr(self):
		self._sendCmd("R?")
		time.sleep(0.05)
		return self._getReply()

	def enable_laser(self, state=False):
		cmd = "ON" if state else "OFF"
		self._sendCmd(cmd)

	def close(self):
		self.led(False)
		try:
			self.ArduinoSerial.close() # close serial port
		except:
			pass
		quit()