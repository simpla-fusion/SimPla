
SET(SAMRAI_DIR /home/salmon/workspace/SAMRAI-master/)
SET(SAMRAI_LIBRARY_PATH ${SAMRAI_DIR}/lib)
SET(SAMRAI_INCLUDE ${SAMRAI_DIR}/source)
LINK_DIRECTORIES(${SAMRAI_LIBRARY_PATH})
LINK_DIRECTORIES(${SAMRAI_DIR}/source/test/testlib)

INCLUDE_DIRECTORIES(${SAMRAI_DIR}/include)
INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE})

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

FILE(GLOB_RECURSE samrai_SRC ./*.cpp)

MESSAGE(${samrai_SRC})
ADD_EXECUTABLE(samrai_mb ${samrai_SRC})

TARGET_LINK_LIBRARIES(samrai_mb  SAMRAI_test ${SAMRAI_LIBRARIES} gfortran   gomp ${MPI_LIBRARIES} ${HDF5_LIBRARIES})
