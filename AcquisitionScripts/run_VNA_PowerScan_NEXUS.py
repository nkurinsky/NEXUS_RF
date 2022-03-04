#!/bin/bash

# How many power scans to perform
n_scans=5

# Power range
pmin=-50.0
pmax=-15.0
pstep=1.0

# Frequency range
# fmin=4.24205e9 ; fmax=4.24225e9	## Al
# fmin=3.93600e9 ; fmax=3.93680e9	## Nb 1
# fmin=3.98280e9 ; fmax=3.98400e9	## Nb 2
# fmin=4.02660e9 ; fmax=4.02760e9	## Nb 3
# fmin=4.07240e9 ; fmax=4.07300e9	## Nb 4
# fmin=4.11720e9 ; fmax=4.11780e9	## Nb 5
fmin=4.20270e9 ; fmax=4.20300e9	## Nb 6
# fmin=4.24470e9 ; fmax=4.24482e9	## Nb 7 (next to resonator)
# fmin=4.29080e9 ; fmax=4.29140e9	## Nb 8
# fmin=4.33050e9 ; fmax=4.33140e9	## Nb 9
# fmin=4.37460e9 ; fmax=4.37630e9	## Nb 10



ns=50000

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
