#- Find SAMRAI 
#  This module will define the following variable:
#  SAMRAI_DIR
#  SAMRAI_CONFIG amrelliptic2d.Linux.64.mpicxx.gfortran.DEBUG.OPT.MPI
# OUTPUT:
#  SAMRAI_FOUND
#  SAMRAI_DIR
#  SAMRAI_INCLUDE_DIR
#  SAMRAI_LIBRARY_DIR
#  SAMRAI_LIBRARIES

if (SAMRAI_INLCUDE_DIRS AND SAMRAI_LIBRARIES)
    #DO NOTHING

else (SAMRAI_INLCUDE_DIRS AND SAMRAI_LIBRARIES)

    SET(SAMRAI_LIBRARIES_
            # SAMRAI_testlib
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


    find_path(SAMRAI_INCLUDE_CONFIG_DIR_
            NAMES SAMRAI/SAMRAI_config.h
            HINTS /usr/include ${SAMRAI_SOURCE_DIR}/source/ ${SAMRAI_BUILD_DIR}/include/
            NO_DEFAULT_PATH
            )


    SET(SAMRAI_INCLUDE_DIRS
            ${SAMRAI_INCLUDE_CONFIG_DIR_}
            ${SAMRAI_SOURCE_DIR}/source/
            )

    FOREACH (LIB_ ${SAMRAI_LIBRARIES_})
        find_library(SAMRAI_LIBRARY_${LIB_} NAMES ${LIB_} PATHS ${SAMRAI_BUILD_DIR})
        LIST(APPEND SAMRAI_LIBRARIES "${SAMRAI_LIBRARY_${LIB_}}")
    ENDFOREACH ()

    LIST(APPEND SAMRAI_LIBRARIES gfortran)

    MARK_AS_ADVANCED(
            SAMRAI_INCLUDE_DIRS
            SAMRAI_LIBRARIES
    )
endif (SAMRAI_INLCUDE_DIRS AND SAMRAI_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SAMRAI DEFAULT_MSG SAMRAI_LIBRARIES)
