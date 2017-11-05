SIMPla {#mainpage}
========================================

\b SIMPla is a unified and hierarchical development framework for plasma simulation. “SimPla” is abbreviation of four words,  __Simulation__, __Integration__, __Multi-physics__ and __Plasma__.

 \note SIMPla is a  [GAPS](http://wiki.gaps.org.cn) project.


  Requirement:
  --------------
  - C++ 11,
    - Expression Template (Field, nTuple), need  gcc >=5
    - others,  need gcc >= 4.8
  - MPI , OpenMPI >1.10.2
  - HDF5  >1.8.10
  - boost : uuid
  - TBB for multi-block_dims
  - Lua >5.2 , for lua_parser
  - CMake 3.5.1, for building


  Optional:
  --------------
  - google test , for unit test
  - CUDA 8, for sp_lite
  - liboce >0.17 for modeling
  - h5py, matplotlib, to display result

  Build:
  --------------

    $mkdir build
    $cd build
    $cmake ../<src make_dir>/ -DCMAKE_BUILD_TYPE=Release
    $make <target>



# Document {#detail}
========================================

 - @subpage install
 - @subpage general_conversions
 - @subpage design_document
 - @subpage user_guide
  
  TODO
  - AMR
  - Embedding boundary
  - Physical quantity/units


