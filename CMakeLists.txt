cmake_minimum_required(VERSION 2.8)

PROJECT(SimPla CXX C)

INCLUDE(CTest)
ENABLE_TESTING()

INCLUDE(ExternalProject)

include(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)

SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(BUILD_SHARED_LIBS OFF)

SET(PREFIX ${CMAKE_CXX_COMPILER_ID}-${CMAKE_BUILD_TYPE})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})


MESSAGE(STATUS "Using C++ Compiler ${CMAKE_CXX_COMPILER_ID} ")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "-std=c++11  -fcolor-diagnostics   -ftemplate-backtrace-limit=0   ")
    ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX)
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    SET(CMAKE_CXX_FLAGS "-std=c++11  -openmp ")
    ADD_DEFINITIONS(-DUSE_MPI -DMPICH_SKIP_MPICXX)
else()
    SET(CMAKE_CXX_FLAGS "  -std=c++11  -openmp  ")
    ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX)

endif ()


FIND_PACKAGE(Lua)
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})


FIND_PACKAGE(MPI)
IF (MPI_FOUND)
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
    ADD_DEFINITIONS(-DUSE_MPI -DMPICH_SKIP_MPICXX)
ENDIF (MPI_FOUND)

SET(HDF5_USE_STATIC_LIBRARIES OFF)
FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL)
IF (HDF5_FOUND)
    INCLUDE_DIRECTORIES(BEFORE ${HDF5_INCLUDE_DIRS})
    IF (HDF5_IS_PARALLEL)
        ADD_DEFINITIONS(-DUSE_PARALLEL_IO)
    ENDIF (HDF5_IS_PARALLEL)
    #LINK_DIRECTORIES(${HDF5_LIBRARY_DIRS})
ENDIF (HDF5_FOUND)

SET(Boost_USE_STATIC_LIBS ON) # only find static libs
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost)

if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else()
    set(boost_cmds
            CONFIGURE_COMMAND ./bootstrap.sh --prefix=<INSTALL_DIR>
            BUILD_COMMAND ./b2 address-model=${am} ${boost_with_args}
            INSTALL_COMMAND ./b2 address-model=${am} ${boost_with_args}
            install
    )
    ExternalProject_Add(boost
            PREFIX ${PROJECT_BINARY_DIR}/external_project
            URL http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.bz2/download
            DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/external_project/src/
            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/boost
            BUILD_IN_SOURCE 1
            INSTALL_COMMAND "")
    ExternalProject_Get_Property(boost install_dir)
    set(BOOST_ROOT "${install_dir}" CACHE INTERNAL "")
    find_package(Boost 1.58.0)
endif ()


#IF(GTEST_ROOT)
#  INCLUDE(${GTEST_ROOT})
#  INCLUDE_DIRECTORIES(${GTEST_ROOT}/include)
#ELSE()
ExternalProject_Add(googletest
        PREFIX ${PROJECT_BINARY_DIR}/external_project
#  URL   http://googletest.googlecode.com/files/gtest-1.7.0.zip
        URL file://${CMAKE_SOURCE_DIR}/external_project/src/gtest-1.7.0.zip
        SOURCE_DIR ${PROJECT_BINARY_DIR}/external_project/googletest
        INSTALL_COMMAND "")
ExternalProject_Get_Property(googletest SOURCE_DIR)
ExternalProject_Get_Property(googletest BINARY_DIR)
INCLUDE_DIRECTORIES(${SOURCE_DIR}/include)
LINK_DIRECTORIES(${BINARY_DIR})
#ENDIF()


#FIND_PACKAGE(CUDA 7.0)


#SET(TBB_ARCH_PLATFORM intel64)
#FIND_PACKAGE(TBB )
#IF(TBB_FOUND)
#	ADD_DEFINITIONS(-DUSE_TBB )
#	INCLUDE_DIRECTORIES(  ${TBB_INCLUDE_DIR} )
#ENDIF( )

execute_process(COMMAND git describe --all --dirty --long
        OUTPUT_VARIABLE PROJECT_IDENTIFY
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

MESSAGE(STATUS "Identify: ${PROJECT_IDENTIFY}")

ADD_DEFINITIONS(-DIDENTIFY=\"${PROJECT_IDENTIFY}\")
#ADD_DEFINITIONS(-Dsimpla=simpla )

# workaround for CLion 1.0.x
FILE(GLOB_RECURSE clion_all_headers ${CMAKE_SOURCE_DIR}/*.h)
add_executable(all_clion ${clion_all_headers} dummy.h dummy.cpp)

add_subdirectory(core)
add_subdirectory(applications)
add_subdirectory(example)
add_subdirectory(docs)

 