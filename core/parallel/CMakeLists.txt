
my_test(distributed_array_test    
         update_ghosts.cpp 
      )
target_link_libraries(distributed_array_test parallel  utilities  )

add_library(parallel  parallel.cpp mpi_datatype.cpp   
  mpi_aux_functions.cpp  mpi_comm.cpp  )
target_link_libraries(parallel ${MPI_LIBRARIES} )

my_test(multi_thread_test    
         multi_thread_test.cpp  

      )
target_link_libraries(multi_thread_test ${TBB_LIBRARIES} )
