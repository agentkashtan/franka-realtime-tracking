#!/bin/bash
set -v
set -e
sudo apt update

sudo apt-get install -y libopenexr-dev
sudo apt install -y build-essential
sudo apt install -y coinor-libipopt-dev
pip3 install casadi

