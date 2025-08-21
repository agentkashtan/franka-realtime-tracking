#!/bin/bash
set -v
date 
sudo ip route del 129.97.71.51 dev wwan0
sleep 3
date 
sudo ip route add 129.97.71.51 dev wwan0
sleep 3
