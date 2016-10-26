
#FIND_PACKAGE(SAMRAI)

#IF (NOT SAMRAI_FOUND)

#MESSAGE(STATUS "SAMRAI is not found. Build it!")

INCLUDE(ExternalProject)

IF (NOT SAMRAI_SRC_URL)
    SET(SAMRAI_SRC_URL https://github.com/LLNL/SAMRAI/archive/master.zip)
ENDIF (NOT SAMRAI_SRC_URL)

SET(Boost_DIR ${Boost_INCLUDE_DIR}/../)

GET_FILENAME_COMPONENT(MPI_BIN_DIR ${MPI_C_COMPILER} DIRECTORY)
SET(MPI_DIR ${MPI_BIN_DIR}/../)

if (IS_DIRECTORY /usr/lib/x86_64-linux-gnu/hdf5/serial/)
    SET(HDF5_DIR /usr/lib/x86_64-linux-gnu/hdf5/serial/)
endif (IS_DIRECTORY /usr/lib/x86_64-linux-gnu/hdf5/serial/)


ExternalProject_Add(SAMRAI
        PREFIX ${PROJECT_BINARY_DIR}/external_project/SAMRAI
        #        URL
        URL ${SAMRAI_SRC_URL}

        CONFIGURE_COMMAND ${cmake} <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
        --with-boost=${Boost_DIR} --with-hdf5=${HDF5_DIR} --with-mpi=${MPI_DIR}

        LOG_DOWNLOAD 1
        LOG_UPDATE 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_TEST 1
        LOG_INSTALL 1
        )
ExternalProject_Get_Property(SAMRAI SOURCE_DIR)
ExternalProject_Get_Property(SAMRAI BINARY_DIR)
SET(SAMRAI_SOURCE_DIR ${SOURCE_DIR})
SET(SAMRAI_BINARY_DIR ${BINARY_DIR})


SET(SAMRAI_INCLUDE_DIRS ${SAMRAI_BINARY_DIR}/include/ ${SAMRAI_SOURCE_DIR}/source/)

SET(SAMRAI_LIBRARIES
        SAMRAI_appu
        SAMRAI_algs
        SAMRAI_solv
        SAMRAI_geom
        SAMRAI_mesh
        SAMRAI_math
        SAMRAI_pdat
        SAMRAI_xfer
        SAMRAI_hier
        SAMRAI_tbox
        gfortran
        )
LINK_DIRECTORIES(${SAMRAI_BINARY_DIR}/lib)

MESSAGE(STATUS "SAMRAI is builded:" libs: ${SAMRAI_LIBRARIES} ;include:${SAMRAI_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})

FILE(GLOB SAMRAIWrapper_SRC ./*.cpp)
add_library(samrai_wrapper ${SAMRAIWrapper_SRC})
TARGET_LINK_LIBRARIES(samrai_wrapper ${SAMRAI_LIBRARIES})
SET(MESH_WEAPPER samrai_wrapper)
add_dependencies(samrai_wrapper SAMRAI)
