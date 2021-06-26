#!/bin/bash

hostIP="192.168.0.34"
port=11034
rto=1

(echo $1; sleep $rto) | nc $hostIP $port -q $rto
