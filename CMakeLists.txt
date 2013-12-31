cmake_minimum_required(VERSION 2.8.5)

PROJECT(SimPla  CXX C)

INCLUDE(CTest)
ENABLE_TESTING()

include(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)
 
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_BUILD_TYPE Release)
#SET(CMAKE_BUILD_TYPE Debug)


INCLUDE(ExternalProject)
ExternalProject_Add(googletest
    PREFIX third_part 
    #    URL   http://googletest.googlecode.com/files/gtest-1.6.0.zip
    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/gtest-1.7.0.zip
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/googletest
    INSTALL_COMMAND "" )
ExternalProject_Get_Property(googletest SOURCE_DIR)
ExternalProject_Get_Property(googletest BINARY_DIR)
INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
LINK_DIRECTORIES(${BINARY_DIR})


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

 
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

SET(CMAKE_CXX_FLAGS "-fPIC -std=c++0x ")
#-ftemplate-backtrace-limit=0
SET(CMAKE_CXX_FLAGS_DEBUG " -DDEBUG -g")
SET(CMAKE_CXX_FLAGS_RELEASE "  -DNDEBUG -O3  ")

if(USE_CLANG)
  SET(CMAKE_AR      "/usr/bin/llvm-ar")
  SET(CMAKE_LINKER  "/usr/bin/llvm-ld")
  SET(CMAKE_NM      "/usr/bin/llvm-nm")
  SET(CMAKE_OBJDUMP "/usr/bin/llvm-objdump")
  SET(CMAKE_RANLIB  "/usr/bin/llvm-ranlib")
  #SET(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} "-stdlib=libc++")

endif(USE_CLANG)

#set(Boost_USE_STATIC_LIBS       OFF)
#set(Boost_USE_MULTITHREADED     OFF)
#set(Boost_USE_STATIC_RUNTIME    OFF)

#FIND_PACKAGE(Boost  COMPONENTS date_time program_options )
#INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

find_package(Lua52)
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})


find_package(MPI)
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})

find_package(HDF5)

#FIND_PACKAGE(XDMF)
#MESSAGE(STATUS "XDMF Install Dir"${XDMF_INSTALL_DIR}) 

execute_process(COMMAND  git describe --all  --dirty --long
            OUTPUT_VARIABLE PROJECT_IDENTIFY      
            WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR} 
            OUTPUT_STRIP_TRAILING_WHITESPACE 
     )

MESSAGE(STATUS "Identify: ${PROJECT_IDENTIFY}")
 
ADD_DEFINITIONS(-DIDENTIFY=\"${PROJECT_IDENTIFY}\" )

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)



add_subdirectory(src )
add_subdirectory(applications )
add_subdirectory(doc )


 
