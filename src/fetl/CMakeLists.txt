my_test(fetl_test_Cartesian   fetl_test_Cartesian.cpp   )
target_link_libraries(fetl_test_Cartesian physics io ${MPI_LIBRARIES} )
my_test(fetl_test_Cartesian_kz   fetl_test_Cartesian_kz.cpp   )
target_link_libraries(fetl_test_Cartesian_kz physics io ${MPI_LIBRARIES} )

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