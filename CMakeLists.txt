cmake_minimum_required(VERSION 2.8.5)

PROJECT(SimPla  CXX C)

INCLUDE(CTest)
ENABLE_TESTING()

INCLUDE(ExternalProject)
ExternalProject_Add(googletest
    PREFIX third_part 
    #URL   http://googletest.googlecode.com/files/gtest-1.6.0.zip
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/googletest
    INSTALL_COMMAND "" )
ExternalProject_Get_Property(googletest SOURCE_DIR)
ExternalProject_Get_Property(googletest BINARY_DIR)
INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
LINK_DIRECTORIES(${BINARY_DIR})

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

 
include(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)
 
SET(CMAKE_VERBOSE_MAKEFILE ON)
#SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Debug)

SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

FIND_PACKAGE(OpenMP)
SET(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} -fPIC -DGTEST_HAS_TR1_TUPLE=0  -std=c++0x ")
SET(CMAKE_CXX_FLAGS_DEBUG " -DDEBUG -g")
SET(CMAKE_CXX_FLAGS_RELEASE "  -DNDEBUG -O3  ")

set(Boost_USE_STATIC_LIBS       OFF)
set(Boost_USE_MULTITHREADED     OFF)
set(Boost_USE_STATIC_RUNTIME    OFF)

FIND_PACKAGE(Boost  COMPONENTS date_time program_options )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

find_package(Lua51)

#FIND_PACKAGE(XDMF)
#MESSAGE(STATUS "XDMF Install Dir"${XDMF_INSTALL_DIR}) 

INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})

execute_process(COMMAND  git describe --all  --dirty --long
            OUTPUT_VARIABLE PROJECT_IDENTIFY      
            WORKING_DIRECTORY   ${PROJECT_SOURCE_DIR} 
            OUTPUT_STRIP_TRAILING_WHITESPACE 
     )

MESSAGE(STATUS "Identify: ${PROJECT_IDENTIFY}")
 
ADD_DEFINITIONS(-DIDENTIFY=\"${PROJECT_IDENTIFY}\" )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src)



add_subdirectory(src )
add_subdirectory(applications )
add_subdirectory(doc )


 
