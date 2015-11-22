SET(PARALLEL_SRC parallel_policy.cpp mpi_datatype.cpp
        mpi_aux_functions.cpp mpi_comm.cpp mpi_update.cpp distributed_object.cpp distributed_array.cpp)
add_library(parallel ${PARALLEL_SRC})

IF (TBB_FOUND)
    target_link_libraries(parallel ${TBB_LIBRARIES})
ENDIF ()

ADD_SUBDIRECTORY(test)
