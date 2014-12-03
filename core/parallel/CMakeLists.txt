
add_library(parallel  parallel.cpp distributed_array.cpp mpi_datatype.cpp
   mpi_aux_functions.cpp  mpi_comm.cpp  )
target_link_libraries(parallel ${MPI_LIBRARIES} )

ADD_SUBDIRECTORY( test)
