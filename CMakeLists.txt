cmake_minimum_required(VERSION 3.2)
# cmake_policy(SET CMP0057 NEW)
cmake_policy(SET CMP0074 NEW)

PROJECT(SimPla CXX C)

MESSAGE(STATUS "Using C++ Compiler ${CMAKE_CXX_COMPILER_ID} ")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

MESSAGE("-- ARCHITECTURE: " ${CMAKE_LIBRARY_ARCHITECTURE})

SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(BUILD_SHARED_LIBS OFF)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/src/include/)
# Set the project name to spmhd and specify that it's a C++ project

# find_package(MPI REQUIRED)
# # Find the AMReX package with the specified components
# find_package(AMReX REQUIRED 2D EB)

# Link the AMReX library to the target
# target_link_libraries(spmhd PUBLIC AMReX::amrex_2d)

# add_subdirectory(src/simpla)
# add_subdirectory(test)
# add_subdirectory(docs)
# add_subdirectory(sp_lite)

add_subdirectory(src/)
add_subdirectory(examples/frc)

# # 首先，使用file命令和GLOB选项获取examples目录下所有的子目录
# file(GLOB EXAMPLES_DIRECTORIES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/examples ${CMAKE_CURRENT_SOURCE_DIR}/examples/*)

# # 然后，遍历所有找到的目录
# foreach(DIR ${EXAMPLES_DIRECTORIES})
#   # 检查这是否真的是一个目录
#   if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples/${DIR})
#     # 如果是，使用add_subdirectory添加到项目中
#     add_subdirectory(examples/${DIR})
#   endif()
# endforeach()
#SET(CMAKE_CXX_FLAGS " -fno-limit-debug-info -ftemplate-backtrace-limit=0")

# INCLUDE(local_conf.cmake)
# INCLUDE(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)
# SET(PREFIX ${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_BUILD_TYPE})
# SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
# SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})


# SET(GTEST_ROOT "/usr/src/googletest/googletest/")
# FIND_PACKAGE(GTest REQUIRED)
# MESSAGE("-- Google Test: " ${GTEST_INCLUDE_DIRS})
# ADD_CUSTOM_TARGET(alltest)
# INCLUDE(CTest)
# ENABLE_TESTING()



# if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
#     #    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winconsistent-missing-override -fcolor-diagnostics
#     #    -ftemplate-backtrace-limit=0 -funroll-loops  -fno-limit-debug-info  ")
#     SET(CMAKE_Fortran_COMPILER gfortran)
# elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")

# elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "PGI")
#     SET(CMAKE_LIBRARY_ARCHITECTURE x86_64-linux-gnu)
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w --gnu --gnu_version=603000 ")
#     SET(OpenACC_CXX_FLAGS " -acc -Minfo=accel")
#     ADD_DEFINITIONS(-D__STRICT_ANSI__)
#     ADD_DEFINITIONS(-D__SIZEOF_POINTER__=8)
#     SET(MPI_C_COMPILER /pkg/pgi/linux86-64/2017/mpi/openmpi/bin/mpicc)
#     SET(LUA_ROOT /pkg/lua/5.3.4/pgi-17.4/)
#     SET(LUA_INCLUDE_DIR ${LUA_ROOT}/include)
#     LINK_DIRECTORIES(${LUA_ROOT}/lib)
#     SET(LUA_LIBRARIES lua)
# else ()
#     SET(OpenACC_CXX_FLAGS " -fopenacc -foffload=nvptx-none -foffload=\"-O3\" -O3  ")
# endif ()
# FIND_PACKAGE(Boost REQUIRED)

# FIND_PACKAGE(OpenMP REQUIRED)
# IF (OpenMP_FOUND)
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# ENDIF (OpenMP_FOUND)

# FIND_PACKAGE(MPI REQUIRED)
# IF (MPI_FOUND)
#     ADD_DEFINITIONS(-DOMPI_SKIP_MPICXX)
#     SET(SKIP_MPICXX true)
#     MESSAGE("-- MPI : " ${MPI_C_INCLUDE_PATH})
# ENDIF (MPI_FOUND)
#FIND_PACKAGE(CUDA)
#SET(TBB_ROOT /pkg/tbb/20170604/)
#FIND_PACKAGE(TBB)
# SET(HDF5_PREFER_PARALLEL OFF)
# SET(HDF5_USE_STATIC_LIBRARIES OFF)
# FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL REQUIRED)

# FIND_PACKAGE(Lua 5.3 REQUIRED)
# IF (LUA_FOUND)
#     MESSAGE(STATUS "Lua version: ${LUA_INCLUDE_DIR} ")
# ENDIF (LUA_FOUND)

# if (NOT IS_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/simpla)
#     FILE(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/simpla)
# endif (NOT IS_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/simpla)


# SET(SIMPLA_INITIALIZE_ARRAY_TO_SIGNALING_NAN ON)
# SET(ENABLE_BOUND_CHECK on)
# ADD_DEFINITIONS(-DOMPI_SKIP_MPICXX)

#LINK_DIRECTORIES(/usr/local/cuda/lib64)
