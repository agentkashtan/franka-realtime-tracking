export CMAKE_PREFIX_PATH=/home/oleksii/lib:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/home/oleksii/lib:/home/oleksii/lib/lib:$LD_LIBRARY_PATH

git clone https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/homel/oleksii/lib

make -j$(11)
sudo make install
cd 


wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.gz
tar -xvzf boost_1_81_0.tar.gz
cd boost_1_81_0
./bootstrap.sh --prefix=/home/oleksii/lib
./b2
./b2 install --prefix=/home/oleksii/lib
cd .. 


git clone --recursive https://github.com/stack-of-tasks/pinocchio
cd pinocchio && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/oleksii/lib
make -j11
sudo make install
cd 

git clone --branch poco-1.11.0 https://github.com/pocoproject/poco.git
cd poco
mkdir cmake-build && cd cmake-build
make -j11
sudo make install
cd 

git clone --recursive https://github.com/frankaemika/libfranka
cd libfranka
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/home/oleksii/lib -DEIGEN3_INCLUDE_DIRS=/home/oleksii/lib/include/eigen3 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build .
sudo make install
cd


sudo cp franka_ik_He.cpp /home/oleksii/lib/include

cd test_pin
rm -rf build
mkdir build
cd build 
cmake ..
