#!/bin/bash

## Define the range of scans you want to process
day_min=20220131
tme_min=000000

day_max=20220204
tme_max=000000

day=$day_min

# counter=1
# while [ $day -le $day_max ]
# do
# 	echo "Day" /data/PowerSweeps/VNA/$day
# 	((day++))
# done

for day_path in /data/PowerSweeps/VNA/*
do
	# echo $day_path
	day=(echo $day_path) | grep -E -o '([0-9])+'
	for series in $day_path/*
	do
		sers=(echo $series) | grep -E -o '([0-9])+\_([0-9])+'
		time="${series: -6}"
		echo $sers
		# if [ "$day" -ge "$day_min" ]; then
		# 	if [ "$day" -le "$day_min" ]; then
		# 		if [ "$time" -ge "$tme_min" ]; then
		# 			if [ "$time" -le "$tme_max" ]; then
		# 				echo sers
		# 			fi
		# 		fi
		# 	fi
		# fi
	done
done

