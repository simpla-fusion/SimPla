my_test(fetl_Cartesian_test   fetl_Cartesian_test.cpp   )
target_link_libraries(fetl_Cartesian_test physics io parallel utilities)
my_test(fetl_Cartesian_kz_test   fetl_Cartesian_kz_test.cpp   )
target_link_libraries(fetl_Cartesian_kz_test physics io parallel utilities)

my_test(fetl_Cylindrical_test  fetl_Cylindrical_test.cpp  )
target_link_libraries(fetl_Cylindrical_test physics io parallel utilities)
my_test(fetl_Cylindrical_kz_test  fetl_Cylindrical_kz_test.cpp  )
target_link_libraries(fetl_Cylindrical_kz_test physics io parallel utilities)


my_test(field_test )
target_link_libraries(field_test physics io parallel utilities)

#my_test(equation_test   physics)
#add_executable(ntuple_test   test/ntuple_test.cpp)
#
#add_library(fetl grid.cpp) 	 