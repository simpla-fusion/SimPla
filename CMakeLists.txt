cmake_minimum_required(VERSION 2.8.5)

PROJECT(SimPla  CXX C)

INCLUDE(CTest)
ENABLE_TESTING()

include(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)
 
SET(CMAKE_VERBOSE_MAKEFILE ON)
#SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)

 
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

SET(CMAKE_CXX_FLAGS_DEBUG " -DDEBUG -g  ")
SET(CMAKE_CXX_FLAGS_RELEASE "  -DNDEBUG -O3  ")
 
MESSAGE(STATUS "Using Compiler ${CMAKE_CXX_COMPILER_ID} ")

if(${CMAKE_CXX_COMPILER_ID}  STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS "-fPIC -std=c++11    -ftemplate-backtrace-limit=0   ")

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

#SET(HDF5_USE_STATIC_LIBRARIES TRUE)

FIND_PACKAGE(HDF5 1.8 COMPONENTS HL )
IF(HDF5_FOUND)
  INCLUDE_DIRECTORIES(BEFORE ${HDF5_INCLUDE_DIR} )
  IF(HDF5_IS_PARALLEL)
    ADD_DEFINITIONS(-DUSE_PARALLEL_IO )
  ENDIF(HDF5_IS_PARALLEL)
ENDIF(HDF5_FOUND)

FIND_PACKAGE(MPI)

IF(MPI_FOUND)

  INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
  ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX )

ENDIF(MPI_FOUND)
#FIND_PACKAGE(XDMF)
#MESSAGE(STATUS "XDMF Install Dir"${XDMF_INSTALL_DIR}) 


 
#FIND_PACKAGE(TBB)
IF(TBB_FOUND)
    ADD_DEFINITIONS(-DHAVE_TBB )
    MESSAGE(${TBB_INCLUDE_DIR})
ENDIF(TBB_FOUND)
ADD_DEFINITIONS(-DIDENTIFY=\"${PROJECT_IDENTIFY}\" )
ADD_DEFINITIONS(-DOMPI_SKIP_MPICXX)
#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)



INCLUDE(ExternalProject)


enable_testing()

ExternalProject_Add(googletest
     PREFIX  ${PROJECT_BINARY_DIR}/third_part 
     #    URL   http://googletest.googlecode.com/files/gtest-1.7.0.zip
    URL file://${CMAKE_SOURCE_DIR}/third_part/gtest-1.7.0.zip
     SOURCE_DIR ${PROJECT_BINARY_DIR}/third_part/googletest
     INSTALL_COMMAND "" )
 ExternalProject_Get_Property(googletest SOURCE_DIR)
 ExternalProject_Get_Property(googletest BINARY_DIR)
 INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
 LINK_DIRECTORIES(${BINARY_DIR})

 
#ExternalProject_Add(trilinos
#    PREFIX third_part 
# #http://trilinos.sandia.gov/download/files/trilinos-11.10.2-Source.tar.bz2
#    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/trilinos-11.10.2-Source.tar.bz2
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/trilinos
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(trilinos SOURCE_DIR)
#ExternalProject_Get_Property(trilinos BINARY_DIR)

 
#ExternalProject_Add(tbb
##    PREFIX third_part 
 #http://trilinos.sandia.gov/download/files/trilinos-11.10.2-Source.tar.bz2
#    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/tbb43_20140724oss_src.tgz
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/tbb43_20140724oss
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(tbb SOURCE_DIR)
#ExternalProject_Get_Property(tbb BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})

#SET(TBB_INSTALL_DIR  ${PROJECT_SOURCE_DIR}/third_part/tbb43_20140724oss)


#ExternalProject_Add(thrust
#    PREFIX third_part 
#    URL  http://thrust.googlecode.com/files/thrust-1.5.1.zip
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/thrust
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(thrust SOURCE_DIR)
#ExternalProject_Get_Property(thrust BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})

#ExternalProject_Add(pugixml
#    PREFIX third_part 
#    #    URL   https://pugixml.googlecode.com/files/pugixml-1.2.tar.gz
#    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/pugixml-1.2.tar.gz
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/pugixml
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(pugixml SOURCE_DIR)
#ExternalProject_Get_Property(pugixml BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})



#ExternalProject_Add(Eigen
#    PREFIX third_part 
#    URL  http://bitbucket.org/eigen/eigen/get/3.2.0.tar.bz2
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/Eigen
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(Eigen SOURCE_DIR)
#ExternalProject_Get_Property(Eigen BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})

#ExternalProject_Add(PHG
#    PREFIX third_part 
#    URL  http://lsec.cc.ac.cn/phg/download/phg-0.8.6-20120910.tar.bz2
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/Eigen
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(PHG SOURCE_DIR)
#ExternalProject_Get_Property(PHG BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})

#ExternalProject_Add(Xdmf
#    PREFIX third_part
#    #GIT_REPOSITORY git://paraview.org/Xdmf.git
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/Xdmf
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(Xdmf SOURCE_DIR)
#ExternalProject_Get_Property(Xdmf BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/libsrc ${BINARY_DIR}/libsrc )
#LINK_DIRECTORIES(${BINARY_DIR}/bin)
#SET(XDMF_NO_MPI ON)











SET(EXT_LIBRARIES    ${LUA_LIBRARIES} ${MPI_LIBRARIES} )
 

 execute_process(COMMAND  git describe --all  --dirty --long
            OUTPUT_VARIABLE PROJECT_IDENTIFY      
            WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR} 
            OUTPUT_STRIP_TRAILING_WHITESPACE 
     )

MESSAGE(STATUS "Identify: ${PROJECT_IDENTIFY}")
add_subdirectory(core )
add_subdirectory(applications )
add_subdirectory(example )
add_subdirectory(docs )


 
