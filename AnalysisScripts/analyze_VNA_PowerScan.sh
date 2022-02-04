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

for day in /data/PowerSweeps/VNA/*
do
	echo $day | grep '^\[0-9]+'
done