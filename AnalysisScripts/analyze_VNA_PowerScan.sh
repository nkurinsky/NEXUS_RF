#!/bin/bash

## Define the range of scans you want to process
day_min=20220201
tme_min=000000

day_max=20220203
tme_max=235959

day=$day_min

for day_path in /data/PowerSweeps/VNA/*
do
	# echo $day_path
	day=$(echo $day_path | grep -E -o '([0-9])+')
	# echo $day
	for series in $day_path/*
	do
		sers=$(echo $series | grep -E -o '([0-9])+\_([0-9])+')
		time="${series: -6}"
		
		## Handle the edge cases first
		if [ "$day" = "$day_min" ]; then
			if [ "$time" -ge "$tme_min" ]; then
				python plot_VNA_PowerScan.py -s $sers
				continue
			fi
		fi

		if [ "$day" = "$day_max" ]; then
			if [ "$time" -le "$tme_max" ]; then
				python plot_VNA_PowerScan.py -s $sers
				continue
			fi
		fi

		# Handle the middling cases
		if [ "$day" -gt "$day_min" ]; then
			if [ "$day" -lt "$day_max" ]; then
				python plot_VNA_PowerScan.py -s $sers
				continue
			fi
		fi
	done
done

