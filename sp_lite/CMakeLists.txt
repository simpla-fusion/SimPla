#cmake_minimum_required(VERSION 2.8)
#SET(CMAKE_VERBOSE_MAKEFILE ON)
#
#PROJECT(sp_lite C CXX)
#
#SET(CMAKE_C_COMPILER /usr/bin/gcc-4.9)
#SET(CMAKE_CXX_COMPILER /usr/bin/g++-4.9)
#SET(CMAKE_C_FLAGS "-std=c11 ")
#
#
#SET(MPI_C_COMPILER /pkg/openmpi/1.10.3-gpu/bin/mpicc)
#SET(MPI_CXX_COMPILER /pkg/openmpi/1.10.3-gpu/bin/mpicxx)
#
#FIND_PACKAGE(MPI REQUIRED)
#
#INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
#
#
#SET(HDF5_PREFER_PARALLEL ON)
#SET(HDF5_USE_STATIC_LIBRARIES OFF)
#
#FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL REQUIRED)
#
#INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
#IF (HDF5_IS_PARALLEL)
#    ADD_DEFINITIONS(-DUSE_PARALLEL_IO -DH5_HAVE_PARALLEL)
#ENDIF (HDF5_IS_PARALLEL)

add_subdirectory(detail)

FILE(GLOB sp_lite_SRC ./*.c)

ADD_EXECUTABLE(sp_boris_mt ${sp_lite_SRC})
TARGET_LINK_LIBRARIES(sp_boris_mt sp_capi sp_lite_openmp stdc++ m mpi)
LINK_DIRECTORIES(/home/salmon/workspace/build/SimPla/lib)

IF (CUDA_FOUND)
    ADD_EXECUTABLE(sp_boris_gpu ${sp_lite_SRC})
    TARGET_LINK_LIBRARIES(sp_boris_gpu pthread sp_lite_cuda stdc++ m mpi gomp sp_capi)
ENDIF (CUDA_FOUND)