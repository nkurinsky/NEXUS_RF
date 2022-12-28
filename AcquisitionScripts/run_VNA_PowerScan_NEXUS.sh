#!/bin/bash

# How many power scans to perform
n_scans=1

# Power range
pmin=-50
pmax=-20
pstep=2

# Frequency range
# fmin=4.24195e9 ; fmax=4.24235e9	## Al -- 4.242175 G
# fmin=4.242095e9 ; fmax=4.242265e9	## Al -- 4.242175 G, 170kHz range
fmin=4.242005e9 ; fmax=4.242355e9	## Al -- 4.242175 G, 350kHz range
# fmin=4.241430e9 ; fmax=4.242930e9	## Al -- 4.242175 G, 1.5MHz range
# fmin=3.93600e9 ; fmax=3.93680e9	## Nb 1
# fmin=3.98280e9 ; fmax=3.98400e9	## Nb 2
# fmin=4.02660e9 ; fmax=4.02760e9	## Nb 3
# fmin=4.07240e9 ; fmax=4.07300e9	## Nb 4
# fmin=4.11720e9 ; fmax=4.11780e9	## Nb 5
# fmin=4.20258e9 ; fmax=4.20308e9	## Nb 6 -- 4.20283 G
# fmin=4.20208e9 ; fmax=4.20358e9	## Nb 6 -- 4.20283 G, 1.5 MHz range
# fmin=4.24456e9 ; fmax=4.24496e9	## Nb 7 (next to resonator) -- 4.24476 G
# fmin=4.244675e9 ; fmax=4.244845e9	## Nb 7 (next to resonator) -- 4.24476 G, 170kHz range
# fmin=4.244585e9 ; fmax=4.244935e9	## Nb 7 (next to resonator) -- 4.24476 G, 350kHz range
# fmin=4.244010e9 ; fmax=4.245510e9	## Nb 7 (next to resonator) -- 4.24476 G, 1.5MHz range
# fmin=4.29080e9 ; fmax=4.29140e9	## Nb 8
# fmin=4.33050e9 ; fmax=4.33140e9	## Nb 9
# fmin=4.37460e9 ; fmax=4.37630e9	## Nb 10

# ns=100000
ns=150000

# How many averages to do at each power
na=20

counter=1
while [ $counter -le $n_scans ]
do
	echo "Scan" $counter
	python VNA_PowerScan_NEXUS.py --P0 $pmin --P1 $pmax --Ps $pstep \
		--F0 $fmin --F1 $fmax --Fn $ns --Na $na
	((counter++))
done
