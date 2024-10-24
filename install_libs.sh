#!/bin/bash
set -v
set -e
export CMAKE_PREFIX_PATH=$HOME/lib:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$HOME/lib:$HOME/lib/lib:$LD_LIBRARY_PATH

cd

#
#git clone https://gitlab.com/libeigen/eigen.git
#cd eigen
#mkdir build
#cd build
#cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/lib
#make -j11
#make install
#cd 
#
#
#wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.gz
#tar -xvzf boost_1_81_0.tar.gz
#cd boost_1_81_0
#./bootstrap.sh --prefix=$HOME/lib
#./b2
#./b2 install --prefix=$HOME/lib
#cd .. 


if [ ! -d pinocchio ]; then
	git clone --recursive https://github.com/stack-of-tasks/pinocchio
fi
if [ ! -d pinocchio/build ];then
	cd pinocchio && mkdir build && cd build
	cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/lib -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_TESTING=OFF
	make -j11
	make install
	cd ..
	cd ..
fi 

#git clone --branch poco-1.11.0 https://github.com/pocoproject/poco.git
#cd poco
#mkdir cmake-build && cd cmake-build
#make -j11
#sudo make install
#cd 

if [ ! -d libfranka ]; then
	git clone --recursive -b 0.9.2 https://github.com/frankaemika/libfranka
fi
if [ ! -d libfranka/build ]; then
	cd libfranka
	mkdir build
	cd build
	cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/lib -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
	cmake  --build .
	make install
	cd
fi

if [ ! -d librealsense ]; then
	git clone https://github.com/IntelRealSense/librealsense
fi
if [ ! -d librealsense/build ]; then
	cd librealsense
	mkdir build
	cd build
	cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/lib -DCMAKE_BUILD_TYPE=Release
	cmake --build . -- -j11
	make install
	cd
fi

sudo apt-get install libopenexr-dev


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo  ${SCRIPT_DIR}
cp ${SCRIPT_DIR}/visual_servoing/franka_ik_He.hpp $HOME/lib/include

