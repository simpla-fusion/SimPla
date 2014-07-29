my_test(distributed_array_test    
         distributed_array_test.cpp  distributed_array.cpp
      )
target_link_libraries(distributed_array_test parallel  utilities  )

add_library(parallel  mpi_datatype.cpp  )
target_link_libraries(parallel ${MPI_LIBRARIES} )
