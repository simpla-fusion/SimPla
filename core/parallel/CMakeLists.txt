add_library(parallel parallel.cpp mpi_datatype.cpp
        mpi_aux_functions.cpp mpi_comm.cpp mpi_update.cpp distributed.cpp distributed_array.cpp)
target_link_libraries(parallel ${MPI_LIBRARIES} utilities)

IF (TBB_FOUND)
    target_link_libraries(parallel ${TBB_LIBRARIES})
ENDIF ()

ADD_SUBDIRECTORY(test)
