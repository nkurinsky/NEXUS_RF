#!/bin/bash

while (true)
do
    line=$(tail -n 1 ~/Desktop/scanLog.txt)
    echo -ne "\r"
    echo -ne $line"                "
    sleep 10
done
