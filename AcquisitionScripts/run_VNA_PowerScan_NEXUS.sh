#!/bin/bash

# How many power scans to perform
n_scans=1

# Power range
pmin=-50.0
pmax=-10.0
pstep=1.0

# Frequency range
# fmin=4.239675e9 ; fmax=4.244675e9	## Al -- 4.242175 G
# fmin=3.93600e9 ; fmax=3.93680e9	## Nb 1
# fmin=3.98280e9 ; fmax=3.98400e9	## Nb 2
# fmin=4.02660e9 ; fmax=4.02760e9	## Nb 3
# fmin=4.07240e9 ; fmax=4.07300e9	## Nb 4
# fmin=4.11720e9 ; fmax=4.11780e9	## Nb 5
# fmin=4.10283e9 ; fmax=4.30283e9	## Nb 6 -- 4.20283 G
fmin=4.24226e9 ; fmax=4.24726e9	## Nb 7 (next to resonator) -- 4.24476 G
# fmin=4.29080e9 ; fmax=4.29140e9	## Nb 8
# fmin=4.33050e9 ; fmax=4.33140e9	## Nb 9
# fmin=4.37460e9 ; fmax=4.37630e9	## Nb 10

ns=100000

# How many averages to do at each power
na=30

counter=1
while [ $counter -le $n_scans ]
do
	echo "Scan" $counter
	python VNA_PowerScan_NEXUS.py --P0 $pmin --P1 $pmax --Ps $pstep \
		--F0 $fmin --F1 $fmax --Fn $ns --Na $na
	((counter++))
done
