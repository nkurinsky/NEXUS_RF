import socket
import select
import time
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

    def setPower(self, power):
        self._sendCmd("SOURce:POWer "+str(power)+"\n")
        print("SOURce:POWer "+str(power))
        #power should be in dBm

    def getPower(self):
        power=0
        power=self._sendCmd("SOURce:POWer?"+"\n")
        print("Power is "+str(power)+" dBm")
        return power

    def singleTrigAndWait(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(self.server_address)
        print("Starting frequency sweep and waiting for complete. . .")
        s.sendall("TRIG:SING\n".encode())
        s.sendall("DISP:WIND:TRAC:Y:AUTO\n".encode())
        s.sendall("*OPC?\n".encode())
        opComplete = s.recv(8)
        print("Done. . . ", "("+str(opComplete)+")")
        s.close()
        return

    def takeSweep(self, f_min, f_max, n_step, n_avs, waittime=8.5, ifb=10e3):
        #Set sweep params
        self._sendCmd("SENS:FREQ:STAR "+str(f_min)+"\n")
        self._sendCmd("SENS:FREQ:STOP "+str(f_max)+"\n")
        self._sendCmd("SENS:SWE:POIN "+str(n_step)+"\n")
        self._sendCmd("CALC:PAR:DEF S21\n")
        self._sendCmd("CALC:SEL:FORM POLar\n")
        self._sendCmd("TRIG:SOUR BUS\n")
        self._sendCmd("SENS:BWID "+str(ifb)+"\n")

        #Set averaging settings
        self._sendCmd("TRIG:AVER ON\n")
        self._sendCmd("SENS:AVER ON\n")
        self._sendCmd("SENS:AVER:COUN "+str(n_avs)+"\n")

        #Autoscale GUI Display
        self._sendCmd("DISP:WIND:TRAC:Y:AUTO\n")
        self.singleTrigAndWait()

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

    def timeDomain(self, lapse, f0, npts, ifb=10e3):
        ## Show the time details
        print("Taking time domain trace with:")
        print("- Central F [Hz]:", f0)
        print("-  Number points:", npts)
        print("-  Sampling rate:", ifb)
        print("-   Duration [s]:", lapse)

        ## Set the frequency parameters
        self._sendCmd("SENS:FREQ:STAR "+str(f0)+"\n")
        self._sendCmd("SENS:FREQ:STOP "+str(f0)+"\n")
        self._sendCmd("SENS:SWE:POIN "+str(int(npts))+"\n")
        self._sendCmd("SENS:BWID "+str(ifb)+"\n")

        #Set averaging settings
        self._sendCmd("TRIG:AVER OFF\n")
        self._sendCmd("SENS:AVER OFF\n")

        # self._sendCmd("CALC:PAR:DEF S21\n")
        self._sendCmd("TRIG:SOUR BUS\n")

        #Set up GUI display window
        self._sendCmd("DISP:SPL 6\n") ## top panel, 2 bottom quadrants

        self._sendCmd("CALC1:PAR:DEF S21\n")
        self._sendCmd("CALC1:SEL:FORM MLOG\n")
        self._sendCmd("DISP:WIND1:TRAC:Y:AUTO\n")

        self._sendCmd("CALC2:PAR:DEF S21\n")
        self._sendCmd("CALC2:SEL:FORM POLar\n")

        self._sendCmd("CALC3:PAR:DEF S21\n")
        self._sendCmd("CALC3:SEL:FORM PHASE\n")

        self._sendCmd("DISPlay:UPDate:IMMediate\n")

        data = ""
        tpts = ""
        elapsed = 0

        ## Take a single time domain trace
        while (elapsed < lapse):
            bgn = time.time()
            self.singleTrigAndWait()
            elapsed += (time.time() - bgn)
            print("Live-time elapsed:",elapsed,"seconds")

            ## Pull the data 
            data += self._getData("CALC:TRAC:DATA:FDAT?\n")
            tpts += self._getData("SENS:FREQ:DATA?\n")
        print("Total live-time elapsed:",elapsed,"seconds")

        S21  = str(data).split(',')
        tpts = str(tpts).split(",")
        
        S21_real = S21[::2]
        S21_imag = S21[1::2]

        return tpts, S21_real, S21_imag

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

    def comp2mag(self, real, imag):
        mags = []
        angles = []

        for i in len(real):
            mags.append(math.sqrt(float(real[i])**2+float(imag[i])**2))
            angles.append(math.atan(float(imag[i])/float(real[i])))

        return mags, angles
