#!/bin/bash

# How many power scans to perform
n_scans=1

# Power range
pmin=-50
pmax=-15
pstep=5

# Frequency range
# fmin=4.24195e9 ; fmax=4.24235e9	## Al -- 4.242175 G
# fmin=4.242095e9 ; fmax=4.242265e9	## Al -- 4.242175 G, 170kHz range
# fmin=4.242005e9 ; fmax=4.242355e9	## Al -- 4.242175 G, 350kHz range
# fmin=4.241233e9 ; fmax=4.2427330e9	## Al -- 4.241983 G, 1.5MHz range

# fmin=3.93545e9 ; fmax=3.93695e9	## Nb 1 -- 3.93620 G, 1.5 MHz range
# fmin=3.98250e9 ; fmax=3.98400e9	## Nb 2 -- 3.98325 G, 1.5 MHz range
# fmin=4.02605e9 ; fmax=4.02755e9	## Nb 3 -- 4.02680 G, 1.5 MHz range
# fmin=4.07175e9 ; fmax=4.07325e9	## Nb 4 -- 4.07250 G, 1.5 MHz range
# fmin=4.11650e9 ; fmax=4.11800e9	## Nb 5 -- 4.11725 G, 1.5 MHz range
# fmin=4.20185e9 ; fmax=4.20335e9	## Nb 6 -- 4.20260 G, 1.5 MHz range
# fmin=4.24380e9 ; fmax=4.24530e9	## Nb 7 -- 4.24455 G, 1.5 MHz range
# fmin=4.28995e9 ; fmax=4.29145e9	## Nb 8 -- 4.29070 G, 1.5 MHz range
# fmin=4.32995e9 ; fmax=4.33145e9	## Nb 9 -- 4.33070 G, 1.5 MHz range
fmin=4.37455e9 ; fmax=4.37605e9	## Nb 10-- 4.37530 G, 1.5 MHz range
# fmin=4.05400e9 ; fmax=4.05900e9	## Box mode

# ns=100000
# ns=150000
ns=20000

# How many averages to do at each power
na=10

counter=1
while [ $counter -le $n_scans ]
do
	echo "Scan" $counter
	python VNA_PowerScan_NEXUS.py --P0 $pmin --P1 $pmax --Ps $pstep \
		--F0 $fmin --F1 $fmax --Fn $ns --Na $na
	((counter++))
done
