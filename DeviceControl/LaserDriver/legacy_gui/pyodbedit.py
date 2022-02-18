#pyodbedit.py
#A python function to issue odbedit commands to the MIDAS odb
#These will only work when issued on the same machine as the MIDAS daq (cdms2 by default)

import os

#read from the odb
def read(path):
  return os.popen('odbedit -c \'ls -v \"'+path+'\"\'').read().strip().split('\n')[0]

#write to the odb
def write(path,val):
  return os.popen('odbedit -c \'set \"'+path+'\" '+val+'\'').read().strip().split('\n')[0]

#start a new run
def runstart():
  #return os.popen('odbedit -c "start now"').read().strip() #What does now do?
  return os.popen('odbedit -c start').read().strip()

#stop the current run
def runstop():
  return os.popen('odbedit -c stop').read().strip()
