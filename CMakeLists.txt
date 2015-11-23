cmake_minimum_required(VERSION 2.8)

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

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    SET(CMAKE_CXX_FLAGS "-std=c++11  -fcolor-diagnostics   -ftemplate-backtrace-limit=0   ")
    SET(COMPILER clang-3.6)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    SET(CMAKE_CXX_FLAGS "-std=c++11")
    SET(MPI_C_COMPILER mpiicc)
    SET(MPI_CXX_COMPILER mpiicpc)
    SET(MPI_Fortran_COMPILER mpiifort)
else ()
    SET(CMAKE_CXX_FLAGS "  -std=c++11   ")
endif ()


FIND_PACKAGE(Lua)
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})
include(FindPackageHandleStandardArgs)

IF (NOT MPI_ROOT)
    FIND_PACKAGE(MPI)
ELSE (NOT MPI_ROOT)
    find_path(MPI_INCLUDE_DIRS
            NAMES mpi.h
            HINTS ${MPI_ROOT}/include/
            DOC "MPI header file path"
            )
    SET(MPI_LIBRARY_DIRS ${MPI_ROOT}/lib/)

    find_library(MPI_LIBRARY mpi HINTS ${MPI_LIBRARY_DIRS})

    SET(MPI_C_LIBRARIES ${MPI_LIBRARY})

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(MPI DEFAULT_MSG MPI_C_LIBRARIES MPI_INCLUDE_DIRS MPI_ROOT)
ENDIF (NOT MPI_ROOT)
IF (MPI_FOUND)
    MESSAGE(MPI ${MPI_INCLUDE_DIRS})
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_DIRS})
    ADD_DEFINITIONS(-DUSE_MPI -DMPICH_SKIP_MPICXX)
    ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX)
ENDIF (MPI_FOUND)

IF (ENV{TBB_ROOT})
    SET(TBB_ROOT $ENV{TBB_ROOT})
ENDIF ()
IF (NOT TBB_ROOT)
    FIND_PACKAGE(TBB)
ELSE (NOT TBB_ROOT)
    find_path(TBB_INCLUDE_DIRS
            NAMES tbb/tbb.h
            HINTS ${TBB_ROOT}/include
            DOC "TBB header file path"
            )
    SET(TBB_LIBRARY_DIRS ${TBB_ROOT}/lib/${ARCH}/${COMPILER})

    find_library(TBB_LIBRARY tbb HINTS ${TBB_LIBRARY_DIRS})

    find_library(TBB_MALLOC_LIBRARY tbbmalloc HINTS ${TBB_LIBRARY_DIRS})

    SET(TBB_LIBRARIES ${TBB_LIBRARY} ${TBB_MALLOC_LIBRARY})

    FIND_PACKAGE_HANDLE_STANDARD_ARGS(TBB DEFAULT_MSG TBB_LIBRARIES TBB_INCLUDE_DIRS TBB_ROOT)

ENDIF (NOT TBB_ROOT)

IF (TBB_FOUND)
    ADD_DEFINITIONS(-DUSE_TBB)
    INCLUDE_DIRECTORIES(  ${TBB_INCLUDE_DIRS})
ENDIF (TBB_FOUND)

IF (ENV{HDF5_ROOT})
    SET(HDF5_ROOT $ENV{HDF5_ROOT})
ENDIF ()
IF (NOT HDF5_ROOT)
#    SET(HDF5_USE_STATIC_LIBRARIES OFF)
#    FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL)
ELSE (NOT HDF5_ROOT)
    find_path(HDF5_INCLUDE_DIRS
            NAMES hdf5.h
            HINTS ${HDF5_ROOT}/include
            DOC "HDF5 header file path"
            )
    SET(HDF5_LIBRARY_DIRS ${HDF5_ROOT}/lib)

    find_library(HDF5_LIBRARY hdf5 HINTS ${HDF5_LIBRARY_DIRS})
    find_library(HDF5_hl_LIBRARY hdf5_hl HINTS ${HDF5_LIBRARY_DIRS})

    SET(HDF5_C_LIBRARIES ${HDF5_LIBRARY} ${HDF5_hl_LIBRARY})

    include(FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(HDF5 DEFAULT_MSG HDF5_C_LIBRARIES HDF5_INCLUDE_DIRS HDF5_ROOT)
ENDIF (NOT HDF5_ROOT)
MESSAGE(${HDF5_C_LIBRARIES})
IF (HDF5_FOUND)
    INCLUDE_DIRECTORIES(  ${HDF5_INCLUDE_DIRS})
    IF (HDF5_IS_PARALLEL)
        ADD_DEFINITIONS(-DUSE_PARALLEL_IO)
    ENDIF (HDF5_IS_PARALLEL)
    #LINK_DIRECTORIES(${HDF5_LIBRARY_DIRS})
ENDIF (HDF5_FOUND)
#
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

 