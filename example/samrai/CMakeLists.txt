
LINK_DIRECTORIES(/home/salmon/workspace/SAMRAI-build)
INCLUDE_DIRECTORIES(/home/salmon/workspace/SAMRAI-build/include)
INCLUDE_DIRECTORIES(/home/salmon/workspace/SAMRAI-master/source)

SET(SAMRAI_LIBRARIES
        SAMRAI_testlib
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

TARGET_LINK_LIBRARIES(samrai_mb ${SAMRAI_LIBRARIES} gfortran gomp ${MPI_LIBRARIES} ${HDF5_LIBRARIES})
