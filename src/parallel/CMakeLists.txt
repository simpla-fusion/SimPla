my_test(distributed_array_test    
         distributed_array_test.cpp 
      )
target_link_libraries(distributed_array_test parallel   )

add_library(parallel  mpi_datatype.cpp  )
target_link_libraries(parallel ${MPI_LIBRARIES} )
