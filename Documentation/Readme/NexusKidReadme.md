# NEXUS KID Readme
Dylan Temples -- November 15, 2021

To "compile" this document, use the provided script:
```
sh compile_readme.sh
```
which simply issues the command:
```
pandoc -f markdown -t html -s -o NexusReadmeView.html NexusKidReadme.md --metadata title="NEXUS KID Readme"
```
Then use a web browser to open the file `NexusReadmeView.html`.

## Network Configuration on the KID PC

The NEXUS KID PC (`nexus-arion`) has a static ip address `192.168.0.42` with subnet mask `255.255.255.0`. It can be accessed over the NEXUS LAN only, using a standard ssh connection (i.e., no Kerberos). It runs a VNC virtual desktop environment on port :5914. To create this desktop if the computer is restarted, use the following command:
```
vncserver :14 -localhost -geometry 1800x900 -depth 24
```
To verify the desktop exists, you can use the command `vncserver -list`. This must be done every time the KID PC is rebooted, so you can use the alias `make_vnc` to create it. This command can be issued over a basic CLI ssh session.

## Network Configuration on the NEXUS Network

To access this desktop from the outside world, one must tunnel through the `nexus-network` server. A virtual desktop on port :5904 is used to connect to the virtual desktop on `nexus-arion`. If that desktop has vanished, you can create it with the same command as above, replacing :14 with :04. Once that desktop is running on `nexus-network`, you can open the virtual desktop and connect to the KID PC's virtual desktop.

First, on the virtual desktop :04 on `nexus-network`, open a terminal and ssh into `nexus-arion` forwarding the port 5914:
```
ssh nexus-admin@192.168.0.42 -L 5914:localhost:5914
```
The password for the `nexus-admin` account is SoupRnexus. Once that connection is established, you can open Remmina or Remote Desktop software to connect to `localhost:14` using the password SoupRvnc. This completes the setup of the virtual desktop on `nexus-network` at port :04.

## Accessing the KID PC from the outside world

To connect to the KID PC while not on the Fermilab network, follow these steps.

1. Connect to the Fermilab VPN using your Services account and one-time password.

2. Get a Kerberos ticket: `kinit <user>`

3. Open a terminal and ssh into the `nexus-network` server, forwarding the virtual desktop port:
```
ssh -K nexusadmin@nexus-network.dhcp.fnal.gov -L 5904:localhost:5904
```
note that for this to work, your username must be added to the `.k5login` file in the home directory of `nexusadmin`.

4. With the connection established, open a VNC viewer on your PC and connect to `localhost:04`.


On the first layer of the virtual desktop, another VNC viewer software should be running already connected to `nexus-arion` port 5914. If it is not, follow the instructions in the previous section to establish the virtual desktop connection.

## Checking Connections on NEXUS LAN

From `nexus-arion`, you should be able to `ping` all the network devices on the NEXUS LAN, including the power strip (192.169.0.145), HEMT power supplies (192.168.0.40 and 192.168.0.41), and the RF switch (192.168.0.43). If you do not get a response, ensure the device is powered on.

The USRP (Ettus Research X300) should have an IP address of `192.168.40.2` but it may show up at `192.168.30.2`. It is connected to the gigabit-ethernet PCI card port with adapter ID `enp179s0f0`. It should be configured as follows:
```
enp179s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 9000
        inet 192.168.40.1  netmask 255.255.255.0  broadcast 192.168.40.255
        inet6 fe80::1dc3:5afc:b493:9700  prefixlen 64  scopeid 0x20<link>
        ether f8:f2:1e:9a:f9:7c  txqueuelen 1000  (Ethernet)
```
With these settings you should be able to get a response from `ping 192.168.40.2`. If not, here are some troubleshooting tips.
- The network routes must be set for the adapters on the GbE card. Run the command `sudo /sbin/route -n` to list current routes. You should get the following output:
```
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
0.0.0.0         192.168.40.1    0.0.0.0         UG    20103  0        0 enp179s0f0
169.254.0.0     0.0.0.0         255.255.0.0     U     1000   0        0 enp179s0f0
192.168.0.0     0.0.0.0         255.255.255.0   U     102    0        0 eno1
192.168.40.0    0.0.0.0         255.255.255.0   U     103    0        0 enp179s0f0
192.168.70.0    0.0.0.0         255.255.255.0   U     0      0        0 enp179s0f1
```
Routes can be created using commands such as `sudo /sbin/route add -net 192.168.40.0 netmask 255.255.255.0 dev enp179s0f0`. These routes persist on a machine reboot. Note that the first entry is the "default" route, which is using the GbE connection to the USRP - this is not desired. To remove this route use the command
```
sudo /sbin/route del -net 0.0.0.0 gw 192.168.40.1 netmask 0.0.0.0 dev enp179s0f0
``` 
When you run the `sudo /sbin/route -n" command with the USB LAN connection enabled, you should have an entry like
```
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
0.0.0.0         131.225.192.222 0.0.0.0         UG    20104  0        0 enxa0cec852e5df
```
If it's not there, you should add it.


- Check the Ubuntu firewall settings, it should be enabled (`sudo ufw enable`) and the command `sudo ufw status` should return the following:
```
Status: active

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW       127.0.0.0/8               
22/tcp                     ALLOW       192.168.0.0/16            
Anywhere                   ALLOW       192.168.30.0/24/tcp       
Anywhere                   ALLOW       192.168.40.0/24/tcp       
22/tcp                     REJECT      Anywhere                  
22/tcp (v6)                REJECT      Anywhere (v6)
```
Firewall rules can be added with commands like `sudo ufw allow proto tcp from 192.168.0.0/16 to any port 22`. To disable the firewall (which can only be done if the USB LAN connection is disabled) use the command `sudo ufw disable` and to reset the rules to rebuild the table, use `sudo ufw reset`. These firewall rules persist on a machine reboot.

- The USB ethernet dongle connects to the Fermilab network, which dictates the firewall rules above (ensuring no ssh connections can be established except for the LAN). This connection is disabled by default (when the machine reboots) and can be enabled with the command `usb_lan_up` if you need to see the internet (for updates, etc.). Only enable the USB LAN connection if the ufw is enabled. The connection can be disabled again using the command `usb_lan_down`. These are simple aliases and can be found by `cat ~/.bash_aliases`.


## Getting or Setting the RF Switch Configuration

The RF switch (MiniCircuits ZTRC-4SPDT-A18) comes with a browser-based tool to adjust its network configuration. The file is located at 
```
file:///home/nexus-admin/Downloads/MCL_PTE_Ethernet_Config(X1).HTML
```
There is no password set for the device (if it somehow gets enabled, the default is 1234), so just enter the IP address of th switch and click "Read Current Configuration". Once the fields below populate, you can adjust them and save the new configuration if necessary.

## Connecting to the Laser Driver Arduino

Connect the USB cable from the laser Arduino to the USB signal conditioner board. The other port of the board should be connected to a USB port on the back of the KID PC. Once it is connected, the device should appear at `/dev/ttyUSB0`. 

To start the laser controller GUI, move to the laser code directory:
```
cd ~/NEXUS_RF/DeviceControl/LaserDriver/
```
Then, run the gui (using Python3):
```
python gui_laser_control.py
```
First. click the "Configure Serial Port" button. Then, click the "Roll Call" button until the panel reading "Minion absent" reads "Laser pulser".

If you cannot run the GUI because it cannot find an available port to communicate over, ensure the local user is added to the `dialout` group:
```
sudo usermod -a -G dialout nexus-admin
sudo chmod a+rw /dev/ttyUSB0
```
then restart the machine.

## Operating the RF Switch

The RF switch is on the NEXUS LAN at IP address 192.168.0.43. To control the switch state, commands can be issued directly through a web browser. For example, to set switch "B" to connect common and port 2, the command to enter in the browser is
```
http://192.168.0.43/SETB=1
```

Alternatively, there is a Python (3) GUI that controls the switch state. It operates both switches "A" and "B" simultaneously, such that the KID is either connected to the VNA or the USRP solely, with no possibility of crossed states. To run this code, ensure you are in a Python3 environment, then navigate to `/home/nexus-admin/NEXUS_RF/DeviceControl/RF_Switch/`. To start the GUI, run
```
python start_gui.py
```
which will open a window. 

First, click the "Probe" button to ensure the RF switch communication is active. Once the probe has been issued, you can set the state of the switch to either "VNA" or "USRP" using the buttons. They act as toggles, so once a state is selected, the only option is to change to the opposite state. Since this code issues commands ad-hoc, there is no need for a persistent TCP/IP connection to the device, meaning one can exit the GUI without needing to disconnect in  any way.

## Starting the VNA

The VNA is controlled by the KID PC over a USB connection. The program that runs it is a compiled executable, which lives at `/home/nexus-admin/NEXUS_RF/DeviceControl/VNA/`. There are two executables, the newer version is `CMT_S2VNA_19.1.3_x86_64.appimage`. This can be run from the command line with the command
```
./CMT_S2VNA_19.1.3_x86_64.appimage
```
while in the appropriate directory, or by opening a file browser (`nautilus`) and double clicking the executable file.

## Starting the USRP

The USRP is controlled by the KID PC over a Gigabit ethernet connection, described above. First we want to verify the USRP can be found by the server software. To do this issue the command `uhd_find_devices --args="addr=192.168.40.2"` which should give the following output:
```
-------------------------
-- UHD Device 0
-------------------------
Device Address:
    serial: 31E0018
    addr: 192.168.40.2
    fpga: XG
    name: 
    product: X300
    type: x300
```
Some settings may need to be changed, including maximum memory, and fpga firmware. To update the settings, use the commands
```
sudo sysctl -w net.core.wmem_max=24862979
sudo sysctl -w net.core.rmem_max=24862979
```
To change the firmware on the USRP, use the following command:
```
 uhd_image_loader --args="addr=192.168.40.2,fpga=XG,type=x300"
```
then power-cycle the USRP using the LAN-controlled power strip, or hardware power button.


Next, a USRP server session must be started. To start a server session, navigate to the relevant directory:
```
cd /home/nexus-admin/NEXUS_RF/DeviceControl/GPU_SDR/
```
and issue the command
```
sudo taskset -c 0,1,2,3,4,5,6,7 ./server
```
The code in `/home/nexus-admin/NEXUS_RF/DeviceControl/GPU_SDR/` can be compiled using the `Makefile` to create the server executable:
```
make -j
```
 

You should start a new terminal before doing this since that terminal will be occupied by the server process. Once a server session has been established, scripts that control the USRP can be run. Note that all the scripts need to be run using `python2.7`. Scripts to acquire data with the USRP can be found at `NEXUS_RF/DeviceControl/GPU_SDR/scripts/`.

## Biasing the KID Amplifiers

The KID amplifiers, both the warm amp and the HEMTs, are connected to the rear Keysight power supply, at IP address 192.168.0.40. To control this device, open a web broswer and type the IP address into the address bar, then log in and go to the "Device Control" tab. You should see a rendering of the device screen and hardware buttons. 

To bias up the warm amp, select Output 1, and change the voltage setpoint to 6.0 V. When you turn this output on, it should read "CV" for constant voltage, and sit around 78 mA. 

To bias the HEMT amplifier, first select Output 3. Change the voltage setpoint for this channel to 0.25 V, which should only draw about 0.034 mA after turning on the output. Once this channel has been biased, select Output 2, and change the voltage set point to 0.2 V. We will ramp this voltage up until it is drawing roughly 22 mA of current. This occurs somewhere between 0.9 V and 1.2 V. For instance, on 2021-12-04, it was biased to 0.95 V and initially drew 21.957 mA, then after having run for a while, it dropped to just below 20 mA. 

With all three power supply channels enabled, the KID is ready to be read out.

## Finding the Al KID with the VNA

To ensure the KID is visible on the system, connect it to the VNA using the RF switch GUI at `~/NEXUS_RF/DeviceControl/RF_Switch/start_gui.py` (Python3). Once the connection through the VNA is established, start the VNA gui:
```
cd ~/NEXUS_RF/DeviceControl/VNA/
./CMT_S2VNA_19.1.3_x86_64.appimage
```
After issuing the second command, the GUI should start. To set the driving power, click on "Stimulus" on the menu bar, then select "Power" from the drop down menu and click "Power" from the sub-menu. On the right-hand menu, adjust the power to be -40 (dB). Ensure that the VNA is reading the S21 transmission parameter (chosen in top-left of GUI).

With the VNA frequency scan running, zoom into near 4.242 GHz to locate the aluminum resonator. The baseline of the trace should be around -13.0 dBm and the peak should be near -20.0 dBm. On the 2021-12-04 power scan, at -40 dB of power, the central frequency of the resonator was 4.24213475 GHz and had a Q of 4.3e5.

## Performing a Power Scan of KID with VNA

While the KID amplifiers are biased, and the VNA GUI is open, a power scan can be performed. Navigate to and run the VNA acquisition script:
```
cd ~/NEXUS_RF/AcquisitionScripts/
python VNA_PowerScan_NEXUS.py
```
which again runs in Python3. This will scan over the stimulus powers that are indicated within the code. These values are hard-coded and are not passed as sarguments when running the script. One may also need to change the frequency range of interest. 

As data is taken, it is saved at `/data/PowerSweeps/VNA/YYYYMMDD/YYYYMMDD_HHMMSS/` where `YYYY`, `MM`, `DD`, `HH`, `MM`, `SS`, are the year, month, day, hour, minute, second when the scan was started. Inside this directory, each power of the scan has its own data file, `Psweep_Ppppp_YYYYMMDD_HHMMSS.txt`, where `pppp` is the power (e.g. `-40.0`). 

To analyze this power scan and look at the peak frequency and resonator Q value as a function of stimulus power, first navigate to the analysis scripts:
```
cd ~/NEXUS_RF/AnalysisScripts/
```
and edit the date and series strings to analyze the desired power scan data. Once that's done, run the script:
```
python plot_VNA_PowerScan.py
```
First it will try to locate, then fit, a single resonance in each file (corresponding to a single power). A diagnositic plot will appear for each power that shows the identified resonance. Click to exit each of these plot windows as they appear. Once all the data files have been fit and parameters extracted, the script will display a plot of Peak Frequency vs Power and Resonator Q vs Power.

## Performing a VNA Measurement with the USRP

Start a server session for the USRP using the above instructions. 
