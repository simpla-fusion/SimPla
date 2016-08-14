
Description:
----------------------
  - 1/2/3D
  - Cold Plasma Fluid
  - FDTD EM solver
  - PML

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
    $python check_point.py ./simpla.h5  "/checkpoint/E"  2 -1&

