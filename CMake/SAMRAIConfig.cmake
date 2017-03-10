SET(SAMRAI_FOUND TRUE)
SET(SAMRAI_DIR /pkg/SAMRAI/3.11.0-debug)
SET(SAMRAI_INCLUDE_DIRS ${SAMRAI_DIR}/include)
SET(SAMRAI_LIBRARY_DIRS ${SAMRAI_DIR}/lib)

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
    find_library(SAMRAI_${LIB_} NAMES SAMRAI_${LIB_} PATHS ${SAMRAI_LIBRARY_DIRS})
    IF (NOT SAMRAI_${LIB_}_NOTFOUND)
        LIST(APPEND SAMRAI_LIBRARIES "${SAMRAI_${LIB_}}")
    ENDIF (NOT SAMRAI_${LIB_}_NOTFOUND)
ENDFOREACH ()

LIST(APPEND SAMRAI_LIBRARIES gfortran ${MPI_LIBRARIES} ${HDF5_LIBRARIES})
