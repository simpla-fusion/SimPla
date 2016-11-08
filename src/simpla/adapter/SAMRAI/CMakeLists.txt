SET(SAMRAI_SRC_URL ${PROJECT_SOURCE_DIR}/external_project/SAMRAI)

IF (NOT EXISTS ${SAMRAI_SRC_URL})
    SET(SAMRAI_SRC_URL https://github.com/LLNL/SAMRAI/archive/master.zip)
ENDIF (NOT EXISTS ${SAMRAI_SRC_URL})

SET(Boost_DIR ${Boost_INCLUDE_DIR}/../)
GET_FILENAME_COMPONENT(MPI_BIN_DIR ${MPI_C_COMPILER} DIRECTORY)
SET(MPI_DIR ${MPI_BIN_DIR}/../)

if (IS_DIRECTORY /usr/lib/x86_64-linux-gnu/hdf5/serial/)
    SET(HDF5_DIR /usr/lib/x86_64-linux-gnu/hdf5/serial/)
endif (IS_DIRECTORY /usr/lib/x86_64-linux-gnu/hdf5/serial/)

INCLUDE(ExternalProject)

ExternalProject_Add(SAMRAI
        PREFIX ${PROJECT_BINARY_DIR}/external_project/SAMRAI
        URL ${SAMRAI_SRC_URL}
        CONFIGURE_COMMAND ${cmake} <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
        --with-boost=${Boost_DIR} --with-hdf5=${HDF5_DIR} --with-mpi=${MPI_DIR} --enable-debug=yes
        LOG_DOWNLOAD 1
        LOG_UPDATE 1
        LOG_CONFIGURE 1
        LOG_BUILD 1
        LOG_TEST 1
        LOG_INSTALL 1
        )

ExternalProject_Get_Property(SAMRAI INSTALL_DIR)

SET(SAMRAI_DIR ${INSTALL_DIR})

SET(SAMRAI_LIBRARY_DIRS ${INSTALL_DIR}/lib)

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

        )
#FOREACH (LIB_ ${SAMRAI_LIBRARIES_})
#    find_library(${LIB_} NAMES ${LIB_} PATHS ${SAMRAI_LIBRARY_DIRS})
#    IF (NOT ${LIB_}_NOTFOUND)
#        LIST(APPEND SAMRAI_LIBRARIES "${${LIB_}}")
#    ENDIF (NOT ${LIB_}_NOTFOUND)
#ENDFOREACH ()
#LIST(APPEND SAMRAI_LIBRARIES gfortran)


MESSAGE(STATUS SAMRAI_DIR ${SAMRAI_DIR})


LINK_DIRECTORIES(${SAMRAI_DIR}/lib)

INCLUDE_DIRECTORIES(${SAMRAI_DIR}/include/)

INCLUDE_DIRECTORIES(${SAMRAI_SRC_URL}/source/)

add_library(samrai_wrapper SAMRAITimeIntegrator.cpp SAMRAITimeIntegrator.cpp)

TARGET_LINK_LIBRARIES(samrai_wrapper ${SAMRAI_LIBRARIES})

add_dependencies(samrai_wrapper SAMRAI)


