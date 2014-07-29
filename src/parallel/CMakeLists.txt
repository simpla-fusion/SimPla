my_test(distributed_array_test             distributed_array.cpp 
         distributed_array_test.cpp 
      )
target_link_libraries(distributed_array_test parallel   utilities )

add_library(parallel  mpi_aux_functions.cpp  )
target_link_libraries(parallel ${MPI_LIBRARIES} )
