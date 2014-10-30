SIMPla {#mainpage}
========================================

\b SIMPla is a unified and hierarchical development framework for plasma simulation.
Its long term goal is to provide complete modeling of a fusion device.
“SimPla” is abbreviation of four words,  __Simulation__, __Integration__, __Multi-physics__ and __Plasma__.

# Overview  {#overview}


 


# Detailed Description {#detailed}


 

 
# Building on ShenMa

~~~~~~~~~~~~~{.bash}
$module load cmake/3.0.2 compiler/llvm/3.5.0 hdf5/1.8.10 mpi/openmpi/1.6.3 lua/5.2.3 python/2.7.8
$git clone git@github.com:simpla-fusion/SimPla.git SimPla
$cd SimPla
$mkdir build
$cd build
$CC=clang CXX=clang++ HDF5_ROOT=$HDF5_ROOT MPI_ROOT=$MPI_ROOT  cmake -DCMAKE_BUILD_TYPE=Release ..

$make <exec name>
~~~~~~~~~~~~~

//					" -n<NUM>   \t number of steps\n"
//					" -s<NUM>   \t recorder per <NUM> steps\n"
//					" -o<STRING>\t output directory\n"
//					" -i<STRING>\t configure file \n"
//					" -c,--config <STRING>\t Lua script passed in as string \n"
//					" -t        \t only read and parse input file, but do not process  \n"
//					" -g,--generator   \t generator a demo input script file \n"
//					" -v<NUM>   \t verbose  \n"
//					" -V        \t print version  \n"
//					" -q        \t quiet mode, standard out  \n"
 

