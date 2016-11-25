IF (NOT SAMRAI_DIR)
    SET(SAMRAI_DIR /pkg/SAMRAI/3.11.0-debug)
ENDIF (NOT SAMRAI_DIR)


FIND_PACKAGE(SAMRAI)

INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})


FILE(GLOB amr_test_SRC ./*.cpp)
add_executable(amr_test ${amr_test_SRC})


TARGET_LINK_LIBRARIES(amr_test ${SAMRAI_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_LIBRARIES} concept mesh manifold toolbox data physics simulation)





