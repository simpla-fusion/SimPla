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
make demo_probe_particle 
./bin/demo_probe_particle 
~~~~~~~~~~~~~

simpla usage:
 -n<NUM>   \t number of steps\n
 -s<NUM>   \t recorder per <NUM> steps\n
 -o<STRING>\t output directory\n
 -i<STRING>\t configure file \n
 -c,--config <STRING>\t Lua script passed in as string \n
 -t        \t only read and parse input file, but do not process  \n
 -g,--generator   \t generator a demo input script file \n
 -v<NUM>   \t verbose  \n
 -V        \t print version  \n
 -q        \t quiet mode, standard out  \n
 

