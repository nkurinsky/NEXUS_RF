#!/bin/bash

hostIP="192.168.0.40"
port=5025
rto=1

(echo $1; sleep $rto) | nc $hostIP $port -q $rto
