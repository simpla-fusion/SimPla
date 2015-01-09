SIMPla {#mainpage}
========================================

\b SIMPla is a unified and hierarchical development framework for plasma simulation.
Its long term goal is to provide complete modeling of a fusion device.
“SimPla” is abbreviation of four words,  __Simulation__, __Integration__, __Multi-physics__ and __Plasma__.

# Document {#detailed}

 - @subpage install
 - @subpage general_convertions
 - @subpage design_document
 - @subpage user_guide
 
 
# Building on ShenMa

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

module purge

module load simpla/develop 

git clone https://github.com/simpla-fusion/SimPla.git <SIMPLA DIR>

cd <SIMPLA DIR>

mkdir build

cd build

CC=clang CXX=clang++ HDF5_ROOT=$HDF5_ROOT  MPI_ROOT=$MPI_ROOT cmake -DCMAKE_BUILD_TYPE=Release ..

make demo_probe_particle 

./bin/demo_probe_particle 


simpla usage:

 -n<NUM>    number of steps
 
 -s<NUM>    recorder per <NUM> steps
 
 -o<STRING> output directory
 
 -i<STRING> configure file 
 
 -c,--config <STRING> Lua script passed in as string 
 
 -t         only read and parse input file, but do not process  
 
 -g,--generator    generator a demo input script file 
 
 -v<NUM>    verbose  
 
 -V         print version  
 
 -q         quiet mode, standard out  
 

~~~~~~~~~~~~~