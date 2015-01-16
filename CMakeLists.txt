cmake_minimum_required(VERSION 2.8.5)

PROJECT(SimPla  CXX C)

INCLUDE(CTest)
ENABLE_TESTING()

include(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)
 
SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

#SET(CMAKE_CXX_FLAGS_DEBUG " -DDEBUG -g  ")
#SET(CMAKE_CXX_FLAGS_RELEASE "  -DNDEBUG -O3  ")
 
MESSAGE(STATUS "Using C++ Compiler ${CMAKE_CXX_COMPILER_ID} ")

if(${CMAKE_CXX_COMPILER_ID}  STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS "-fPIC -std=c++11  -fcolor-diagnostics   -ftemplate-backtrace-limit=0   ")

#  SET(CMAKE_AR      "llvm-ar-3.5")
 # SET(CMAKE_LINKER  "llvm-ld-3.5")
  #SET(CMAKE_LINKER_FLAG " -L/usr/lib/gcc/x86_64-linux-gnu/4.8/")
  #SET(CMAKE_NM      "llvm-nm-3.5")
 # SET(CMAKE_OBJDUMP "llvm-objdump-3.5")
  #SET(CMAKE_RANLIB  "llvm-ranlib-3.5")
  #SET(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} "-stdlib=libc++")

elseif(${CMAKE_CXX_COMPILER_ID}  STREQUAL "Intel")
  SET(CMAKE_CXX_FLAGS "-fPIC -std=c++11  -fopenmp  ")
else()
  SET(CMAKE_CXX_FLAGS "-fPIC -std=c++11  -fopenmp -ftemplate-backtrace-limit=0 ")
endif ()

 
FIND_PACKAGE(Lua52)
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})


FIND_PACKAGE(MPI )
IF(MPI_FOUND)
  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
  ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX )
ENDIF(MPI_FOUND)

FIND_PACKAGE(HDF5 1.8 COMPONENTS C  HL )
IF(HDF5_FOUND)
  INCLUDE_DIRECTORIES(BEFORE ${HDF5_INCLUDE_DIRS} )
  IF(HDF5_IS_PARALLEL)
    ADD_DEFINITIONS(-DUSE_PARALLEL_IO )
  ENDIF(HDF5_IS_PARALLEL)
ENDIF(HDF5_FOUND)
 
 
#IF(GTEST_ROOT)
enable_testing()
#INCLUDE(${GTEST_ROOT})
INCLUDE_DIRECTORIES(${GTEST_ROOT}/include)
#ENDIF()


FIND_PACKAGE(CUDA 7.0)


SET(TBB_ARCH_PLATFORM intel64)
FIND_PACKAGE(TBB )
IF(TBB_FOUND)
	ADD_DEFINITIONS(-DUSE_TBB )
	INCLUDE_DIRECTORIES(  ${TBB_INCLUDE_DIR} )
ENDIF( )

execute_process(COMMAND  git describe --all  --dirty --long
            OUTPUT_VARIABLE PROJECT_IDENTIFY      
            WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR} 
            OUTPUT_STRIP_TRAILING_WHITESPACE 
     )

MESSAGE(STATUS "Identify: ${PROJECT_IDENTIFY}")

ADD_DEFINITIONS(-DIDENTIFY=\"${PROJECT_IDENTIFY}\" )

INCLUDE(ExternalProject)
ExternalProject_Add(googletest
     PREFIX  ${PROJECT_BINARY_DIR}/external_project 
     #    URL   http://googletest.googlecode.com/files/gtest-1.7.0.zip
    URL file://${CMAKE_SOURCE_DIR}/external_project/src/gtest-1.7.0.zip
     SOURCE_DIR ${PROJECT_BINARY_DIR}/external_project/googletest
     INSTALL_COMMAND "" )
 ExternalProject_Get_Property(googletest SOURCE_DIR)
 ExternalProject_Get_Property(googletest BINARY_DIR)
 INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
 LINK_DIRECTORIES(${BINARY_DIR})

add_subdirectory(core )
add_subdirectory(applications )
add_subdirectory(example )
add_subdirectory(doc )

 
