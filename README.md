SIMPla {#mainpage}
========================================

\b SIMPla is a unified and hierarchical development framework for plasma simulation.
Its long term goal is to provide complete modeling of a fusion device.
“SimPla” is abbreviation of four words,  __Simulation__, __Integration__, __Multi-physics__ and __Plasma__.

# Overview  {#overview}


 


# Detailed Description {#detailed}


test 

 
# Building on ShenMa

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {.bash}
module purge
module load simpla/develop 
git clone https://github.com/simpla-fusion/SimPla.git <SIMPLA DIR>
cd <SIMPLA DIR>
mkdir build
cd build
CC=clang CXX=clang++ HDF5_ROOT=$HDF5_ROOT  MPI_ROOT=$MPI_ROOT cmake -DCMAKE_BUILD_TYPE=Release ..
make demo_pic
./bin/demo_pic -i ../example/use_case/pic/demo_pic.lua  
 
