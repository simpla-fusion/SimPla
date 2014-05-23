my_test(fetl_test    
#    fetl_test1.cpp 
#    fetl_test2.cpp 
 fetl_test3.cpp
#     fetl_test4.cpp
      )
target_link_libraries(fetl_test physics io ${MPI_LIBRARIES} )

my_test(field_io_test    
  field_io_test.cpp  
      )
target_link_libraries(field_io_test physics io ${MPI_LIBRARIES} )

my_test(ntuple_test   )
target_link_libraries(ntuple_test physics  )
#my_test(equation_test   physics)
#add_executable(ntuple_test   test/ntuple_test.cpp)
#
#add_library(fetl grid.cpp) 	 