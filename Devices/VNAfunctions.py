import socket
import select
from time import sleep
import math

class VNA:

    def __init__(self, server_ip='127.0.0.1', server_port=5025):
        self.server_address = (server_ip, server_port)

    def _sendCmd(self,cmd):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(self.server_address)
        s.sendall(cmd.encode())
        s.close()
        return

    def _getData(self,cmd):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(self.server_address)
        #Clear register
        s.sendall("*CLS\n".encode())
        s.sendall(cmd.encode())
        s.settimeout(120)
        try:
            chunks = []
            data = b''.join(chunks)
            while not "\n" in data.decode():
                chunk = s.recv(4096)
                if chunk == b'':
                    raise RuntimeError("socket connection broken")
                chunks.append(chunk)
                data = b''.join(chunks)

        except socket.timeout:
            s.close()
            raise RuntimeError("No data received from VNA")

        datavals = data.decode()
        datavals = datavals.rstrip("\n")
        return datavals

#    def _waitCmd(self):
#        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#        s.connect(self.server_address)
#        s.sendall("*OPC?\n")
#        opComplete = s.recv(8)
#
#        return

    def setPower(self, power):
        self._sendCmd("SOURce:POWer "+str(power)+"\n")
        print("SOURce:POWer "+str(power))
        #power should be in dBm

    def getPower(self):
        power=0
        power=self._sendCmd("SOURce:POWer?"+"\n")
        print("power is "+str(power)+" dBm")
        return power

    def takeSweep(self, f_min, f_max, n_step, n_avs, waittime=8.5, ifb=10e3):
        #Set sweep params
        self._sendCmd("SENS:FREQ:STAR "+str(f_min)+"\n")
        self._sendCmd("SENS:FREQ:STOP "+str(f_max)+"\n")
        self._sendCmd("SENS:SWE:POIN "+str(n_step)+"\n")
        self._sendCmd("CALC:PAR:DEF S12\n")
        self._sendCmd("CALC:SEL:FORM POLar\n")
        self._sendCmd("TRIG:SOUR BUS\n")
        self._sendCmd("SENS:BWID "+str(ifb)+"\n")

        #Set averaging settings
        self._sendCmd("TRIG:AVER ON\n")
        self._sendCmd("SENS:AVER ON\n")
        self._sendCmd("SENS:AVER:COUN "+str(n_avs)+"\n")

        #trigger sweep
        self._sendCmd("TRIG:SING\n")
        print('now waiting ' + str(waittime*n_avs))
        sleep(waittime*n_avs)
        #probably should implement an actual *OPC? query here but that appears to be broken on Python 3?
        #self._waitCmd()
        #Autoscale GUI Display
        self._sendCmd("DISP:WIND:TRAC:Y:AUTO\n")

        data = self._getData("CALC:TRAC:DATA:FDAT?\n")
        fs = self._getData("SENS:FREQ:DATA?\n")

        freqs=str(fs)
        S21=str(data)
        #S21_phase=str(phase_data

        freqs = freqs.split(",")
        S21 = S21.split(',')
        S21_real = S21[::2]
        S21_imag = S21[1::2]

        #print(freqs)
        #print(S21)
        return freqs, S21_real, S21_imag

    def storeData(self, freqs, S21_real, S21_imag, filename):
        fullname = filename+'.txt'
        f = open(fullname, "a")

        f.write("freq (Hz), S21 Real, S21 Imag\n")

        points=len(freqs)

        i=0
        for i in range(points):
            line=freqs[i]+","+S21_real[i]+","+S21_imag[i]+"\n"
            f.write(line)

        f.close()

        return

    def readData(self, fname):
        freqs = []
        real = []
        imag = []
        f = open(fname, "r")

        #remove header
        buffer = f.readline()

        line = f.readline()
        while len(line)>2:
            line = line.rstrip("\n")
            data = line.split(",")
            freqs.append(float(data[0])/1e6)
            real.append(float(data[1]))
            imag.append(float(data[2]))
            line = f.readline()

        f.close()
        return freqs, real, imag

    def readData_old(self, fname):
        freqs = []
        real = []
        f = open(fname, "r")

        #remove header
        buffer = f.readline()

        line = f.readline()
        while len(line)>2:
            line = line.rstrip("\n")
            data = line.split(",")
            freqs.append(float(data[0])/1e6)
            real.append(float(data[1]))
            line = f.readline()

        f.close()
        return freqs, real

    def comp2mag(self, real, imag):
        mags = []
        angles = []

        for i in len(real):
            mags.append(math.sqrt(float(real[i])**2+float(imag[i])**2))
            angles.append(math.atan(float(imag[i])/float(real[i])))

        return mags, angles
