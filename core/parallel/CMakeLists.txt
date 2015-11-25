SET(PARALLEL_SRC parallel.cpp mpi_datatype.cpp
        mpi_aux_functions.cpp mpi_comm.cpp mpi_update.cpp distributed_object.cpp)
add_library(parallel ${PARALLEL_SRC})

target_link_libraries(parallel ${TBB_LIBRARIES} ${MPI_C_LIBRARIES})


ADD_SUBDIRECTORY(test)
