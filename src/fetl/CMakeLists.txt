my_test(fetl_test_Euclidean   fetl_test_Euclidean.cpp   )
target_link_libraries(fetl_test_Euclidean physics io ${MPI_LIBRARIES} )
my_test(fetl_test_Euclidean_kz   fetl_test_Euclidean_kz.cpp   )
target_link_libraries(fetl_test_Euclidean_kz physics io ${MPI_LIBRARIES} )

my_test(fetl_test_Cylindrical  fetl_test_Cylindrical.cpp  )
target_link_libraries(fetl_test_Cylindrical physics io ${MPI_LIBRARIES} )
my_test(fetl_test_Cylindrical_kz  fetl_test_Cylindrical_kz.cpp  )
target_link_libraries(fetl_test_Cylindrical_kz physics io ${MPI_LIBRARIES} )



my_test(field_io_test    
  field_io_test.cpp  
      )
target_link_libraries(field_io_test physics io ${MPI_LIBRARIES} )


#my_test(equation_test   physics)
#add_executable(ntuple_test   test/ntuple_test.cpp)
#
#add_library(fetl grid.cpp) 	 