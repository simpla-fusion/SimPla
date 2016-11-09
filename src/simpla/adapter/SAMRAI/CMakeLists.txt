
IF (NOT SAMRAI_DIR)
    SET(SAMRAI_DIR /pkg/SAMRAI/3.11.0-debug)
ENDIF (NOT SAMRAI_DIR)

FIND_PACKAGE(SAMRAI)

INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})

add_library(samrai_wrapper  SAMRAITimeIntegrator.cpp)

TARGET_LINK_LIBRARIES(samrai_wrapper ${SAMRAI_LIBRARIES}  ${HDF5_LIBRARIES}  ${MPI_LIBRARIES}  concept mesh toolbox data simulation   )



