
Description:
----------------------
  - cylindrical coordinates
  - Cold Plasma Fluid
  - FDTD EM solver  

Build:
--------------
    $tar xzvf <src>.tar.gz
    $mkdir build
    $cd build
    $cmake ../<src dir>/ -DCMAKE_BUILD_TYPE=Release
    $make demo_em

Usage:
----------------------
    $./demo_em -i demo.lua
    $python CheckPoint.py ./simpla.h5  "/checkpoint/E"  2 -1&

