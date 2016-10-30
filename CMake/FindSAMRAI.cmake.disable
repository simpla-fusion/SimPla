#- Find SAMRAI 
#  This module will define the following variable:
#  SAMRAI_DIR
#  SAMRAI_CONFIG
#  SAMRAI_SOURCE_DIR
# OUTPUT:
#  SAMRAI_FOUND
#  SAMRAI_DIR
#  SAMRAI_INCLUDE_DIR
#  SAMRAI_LIBRARY_DIR
#  SAMRAI_LIBRARIES

IF (NOT SAMRAI_INCLUDE_DIRS)
    find_path(SAMRAI_INCLUDE_CONFIG_DIR_
            NAMES SAMRAI/SAMRAI_config.h
            HINTS /usr/include ${SAMRAI_SOURCE_DIR}/source/ ${SAMRAI_BINARY_DIR}/include/
            NO_DEFAULT_PATH
            )

    IF (SAMRAI_INCLUDE_CONFIG_DIR_FOUND)
        SET(SAMRAI_INCLUDE_DIRS
                ${SAMRAI_INCLUDE_CONFIG_DIR_}
                ${SAMRAI_SOURCE_DIR}/source/
                )
    ENDIF (SAMRAI_INCLUDE_CONFIG_DIR_FOUND)
ENDIF (NOT SAMRAI_INCLUDE_DIRS)

IF (NOT SAMRAI_LIBRARIES)
    SET(SAMRAI_LIBRARIES_
            # SAMRAI_testlib
            appu
            algs
            solv
            geom
            mesh
            math
            pdat
            xfer
            hier
            tbox
            )


    FOREACH (LIB_ ${SAMRAI_LIBRARIES_})
        find_library(SAMRAI_${LIB_} NAMES SAMRAI_${LIB_} PATHS ${SAMRAI_BINARY_DIR}/lib)
        IF (SAMRAI_${LIB_}_FOUND)
            LIST(APPEND SAMRAI_LIBRARIES "${SAMRAI_${LIB_}}")
        ENDIF (SAMRAI_${LIB_}_FOUND)
    ENDFOREACH ()

    LIST(APPEND SAMRAI_LIBRARIES gfortran)
ENDIF (NOT SAMRAI_LIBRARIES)


MARK_AS_ADVANCED(
        SAMRAI_INCLUDE_DIRS
        SAMRAI_LIBRARIES
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SAMRAI
        DEFAULT_MSG SAMRAI_INCLUDE_DIRS SAMRAI_LIBRARIES)
