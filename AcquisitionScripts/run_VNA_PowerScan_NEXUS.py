#!/bin/bash

# How many power scans to perform
n_scans=1

# Power range
pmin=-50.0
pmax=-20.0
pstep=1.0

# Frequency range
fmin=4.24205e9
fmax=4.24225e9
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