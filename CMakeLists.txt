cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0028 NEW)

PROJECT(SimPla CXX C)

INCLUDE(CTest)
ENABLE_TESTING()

INCLUDE(ExternalProject)

INCLUDE(${PROJECT_SOURCE_DIR}/CMake/utils.cmake)

SET(CMAKE_VERBOSE_MAKEFILE ON)

SET(BUILD_SHARED_LIBS OFF)

SET(PREFIX ${CMAKE_CXX_COMPILER_ID}-${CMAKE_BUILD_TYPE})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

SET(ARCH intel64)

MESSAGE(STATUS "Using C++ Compiler ${CMAKE_CXX_COMPILER_ID} ")

SET(COMPILER $ENV{PRG_ENV})

SET(CMAKE_C_FLAGS "-std=c11")


if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "-std=c++11  -fcolor-diagnostics   -ftemplate-backtrace-limit=0  -fPIC ")

elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    SET(CMAKE_CXX_FLAGS "-std=c++11")
else ()
    SET(CMAKE_CXX_FLAGS "-std=c++11   ")
endif ()

#include(FindPackageHandleStandardArgs)


###########################################################
#IF (DEFINED ENV{LUA_ROOT})
#
#    set(LUA_ROOT $ENV{LUA_ROOT})
#
#    find_path(LUA_INCLUDE_DIR
#            NAMES lua.h
#            HINTS ${MPI_ROOT}/include/
#            DOC "MPI header file path"
#            )
#    SET(LUA_LIBRARY_DIRS ${LUA_ROOT}/lib/)
#
#    find_library(LUA_LIBRARY liblua.a HINTS ${LUA_LIBRARY_DIRS})
#
#    SET(LUA_LIBRARIES ${LUA_LIBRARY})
#
#    FIND_PACKAGE_HANDLE_STANDARD_ARGS(LUA DEFAULT_MSG LUA_LIBRARIES LUA_INCLUDE_DIRS LUA_ROOT)
#ELSE (DEFINED ENV{LUA_ROOT})
FIND_PACKAGE(Lua)
#ENDIF (DEFINED ENV{LUA_ROOT})

IF (LUA_FOUND)
    INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})
ENDIF (LUA_FOUND)

###########################################################
#IF (DEFINED ENV{MPI_ROOT})
#    set(MPI_ROOT $ENV{MPI_ROOT})
#    find_path(MPI_INCLUDE_DIRS
#            NAMES mpi.h
#            HINTS ${MPI_ROOT}/include/
#            DOC "MPI header file path"
#            )
#    SET(MPI_LIBRARY_DIRS ${MPI_ROOT}/lib/)
#
#    find_library(MPI_LIBRARY mpi HINTS ${MPI_LIBRARY_DIRS})
#
#    SET(MPI_C_LIBRARIES ${MPI_LIBRARY})
#
#    FIND_PACKAGE_HANDLE_STANDARD_ARGS(MPI DEFAULT_MSG MPI_C_LIBRARIES MPI_INCLUDE_DIRS MPI_ROOT)
#
#    mark_as_advanced(MPI_C_LIBRARIES)
#
#ELSE (DEFINED ENV{MPI_ROOT})
#    SET(MPI_ROOT /usr/lib/openmpi)
FIND_PACKAGE(MPI)
#ENDIF (DEFINED ENV{MPI_ROOT})

IF (MPI_FOUND)
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
    ADD_DEFINITIONS(-DUSE_MPI -DMPICH_SKIP_MPICXX)
    ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX)
ENDIF (MPI_FOUND)

###########################################################
#IF (DEFINED ENV{TBB_ROOT})
#    set(TBB_ROOT $ENV{TBB_ROOT})
#
#    find_path(TBB_INCLUDE_DIRS
#            NAMES tbb/tbb.h
#            HINTS ${TBB_ROOT}/include
#            DOC "TBB header file path"
#            )
#
#    set(TBB_LIBRARY_DIRS $ENV{TBB_LIBRARY_DIRS})
#
#    find_library(TBB_LIBRARY tbb HINTS ${TBB_ROOT}/lib/${ARCH}/${COMPILER})
#
#    find_library(TBB_MALLOC_LIBRARY tbbmalloc ${TBB_ROOT}/lib/${ARCH}/${COMPILER})
#
#
#    FIND_PACKAGE_HANDLE_STANDARD_ARGS(TBB DEFAULT_MSG TBB_LIBRARIES TBB_INCLUDE_DIRS TBB_ROOT)
#
#    set(TBB_FOUND TRUE)
#
#
#ELSE (DEFINED ENV{TBB_ROOT})
#    SET(TBB_ROOT /usr/local)
FIND_PACKAGE(TBB)
#    SET(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_MALLOC_LIBRARY})

#ENDIF (DEFINED ENV{TBB_ROOT})

IF (TBB_FOUND)

    ADD_DEFINITIONS(-DUSE_TBB)

    INCLUDE_DIRECTORIES(${TBB_INCLUDE_DIRS})


ENDIF (TBB_FOUND)
###########################################################
#
#IF (DEFINED ENV{HDF5_ROOT})
#
#    SET(HDF5_ROOT $ENV{HDF5_ROOT})
#
#    find_path(HDF5_INCLUDE_DIRS
#            NAMES hdf5.h
#            HINTS ${HDF5_ROOT}/include
#            DOC "HDF5 header file path"
#            )
#    SET(HDF5_LIBRARY_DIRS ${HDF5_ROOT}/lib)
#
#    find_library(HDF5_LIBRARY hdf5 HINTS ${HDF5_LIBRARY_DIRS})
#
#    find_library(HDF5_hl_LIBRARY hdf5_hl HINTS ${HDF5_LIBRARY_DIRS})
#
#    SET(HDF5_C_LIBRARIES z dl m ${HDF5_LIBRARY} ${HDF5_hl_LIBRARY})
#
#    SET(HDF5_IS_PARALLEL true)
#
#    include(FindPackageHandleStandardArgs)
#
#    FIND_PACKAGE_HANDLE_STANDARD_ARGS(HDF5 DEFAULT_MSG HDF5_C_LIBRARIES HDF5_INCLUDE_DIRS HDF5_ROOT)
#
#ELSE (DEFINED ENV{HDF5_ROOT})

SET(HDF5_PREFER_PARALLEL ON)
SET(HDF5_USE_STATIC_LIBRARIES OFF)

FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL)

#ENDIF (DEFINED ENV{HDF5_ROOT})

IF (HDF5_FOUND)
    INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
    IF (HDF5_IS_PARALLEL)
        ADD_DEFINITIONS(-DUSE_PARALLEL_IO -DH5_HAVE_PARALLEL)
    ENDIF (HDF5_IS_PARALLEL)
ENDIF (HDF5_FOUND)
ADD_DEFINITIONS(-DNO_XDMF)


#IF (DEFINED ENV{XDMF_ROOT})
#
#    SET(XDMF_ROOT $ENV{XDMF_ROOT})
#
#    find_path(XDMF_INCLUDE_DIRS
#            NAMES Xdmf.h
#            HINTS ${XDMF_ROOT}/include
#            DOC "XDMF header file path"
#            )
#    SET(XDMF_LIBRARY_DIRS ${HDF5_ROOT}/lib)
#
#    find_library(XDMF_LIBRARY xdmf HINTS ${XDMF_LIBRARY_DIRS})
#
#
#    SET(XDMF_C_LIBRARIES ${XDMF_LIBRARY})
#
#    include(FindPackageHandleStandardArgs)
#
#    FIND_PACKAGE_HANDLE_STANDARD_ARGS(XDMF DEFAULT_MSG XDMF_C_LIBRARIES XDMF_INCLUDE_DIRS XDMF_ROOT)
##
##ELSE (DEFINED ENV{XDMF_ROOT})
##
##
##    FIND_PACKAGE(XDMF)
##
#ENDIF (DEFINED ENV{XDMF_ROOT})
#
#IF (XDMF_FOUND)
#    INCLUDE_DIRECTORIES(${XDMF_INCLUDE_DIRS})
#
#ENDIF (XDMF_FOUND)

###########################################################

#SET(Boost_USE_STATIC_LIBS ON) # only find static libs
#SET(Boost_USE_MULTITHREADED ON)
#SET(Boost_USE_STATIC_RUNTIME OFF)
#find_package(Boost)
#
#if (Boost_FOUND)
#    include_directories(${Boost_INCLUDE_DIRS})
#else ()
#    set(boost_cmds
#            CONFIGURE_COMMAND ./bootstrap.sh --prefix=<INSTALL_DIR>
#            BUILD_COMMAND ./b2 address-model=${am} ${boost_with_args}
#            INSTALL_COMMAND ./b2 address-model=${am} ${boost_with_args}
#            install
#            )
#    ExternalProject_Add(boost
#            PREFIX ${PROJECT_BINARY_DIR}/external_project
#            URL http://sourceforge.net/projects/boost/files/boost/1.58.0/boost_1_58_0.tar.bz2/download
#            DOWNLOAD_DIR ${CMAKE_SOURCE_DIR}/external_project/src/
#            SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/boost
#            BUILD_IN_SOURCE 1
#            INSTALL_COMMAND "")
#    ExternalProject_Get_Property(boost install_dir)
#    set(BOOST_ROOT "${install_dir}" CACHE INTERNAL "")
#    find_package(Boost 1.58.0)
#endif ()
enable_testing()
FIND_PACKAGE(GTest)

IF (NOT ${GTEST_FOUND})

    ExternalProject_Add(googletest
            PREFIX ${PROJECT_BINARY_DIR}/external_project
            #  URL   http://googletest.googlecode.com/files/gtest-1.7.0.zip
            URL file://${CMAKE_SOURCE_DIR}/external_project/src/gtest-1.7.0.zip
            SOURCE_DIR ${PROJECT_BINARY_DIR}/external_project/googletest
            INSTALL_COMMAND "")
    ExternalProject_Get_Property(googletest GTEST_ROOT)
    ExternalProject_Get_Property(googletest GTEST_ROOT_BINARY_DIR)

    LINK_DIRECTORIES(${BINARY_DIR})
ENDIF (NOT ${GTEST_FOUND})

FIND_PACKAGE(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

#IF (USE_CUDA)
FIND_PACKAGE(CUDA 7.5)
IF (CUDA_FOUND)
    ADD_DEFINITIONS(-DUSE_CUDA)
ELSE (CUDA_FOUND)
    UNSET(USE_CUDA)
ENDIF (CUDA_FOUND)
#ENDIF (USE_CUDA)

execute_process(COMMAND git describe --all --dirty --long
        OUTPUT_VARIABLE PROJECT_IDENTIFY
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

MESSAGE(STATUS "Identify: ${PROJECT_IDENTIFY}")

ADD_DEFINITIONS(-DIDENTIFY=\"${PROJECT_IDENTIFY}\")

# workaround for CLion 1.0.x
FILE(GLOB_RECURSE clion_all_headers ${CMAKE_SOURCE_DIR}/*.h)
add_executable(all_clion ${clion_all_headers} src/dummy.h src/dummy.cpp tools/geqdsk_convert.cpp)

ADD_CUSTOM_TARGET(alltest)


add_subdirectory(src)
add_subdirectory(sp_lite)
add_subdirectory(example)
add_subdirectory(tools)
add_subdirectory(docs)
#add_subdirectory(plugins)
#add_subdirectory(applications)

 
