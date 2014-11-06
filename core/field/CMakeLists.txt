
my_test(field_io_test     field_io_test.cpp        )
target_link_libraries(field_io_test physics io parallel utilities)

 

simpla_test(dummy_manifold_test    field_basic_algerbra_test.cpp)
target_link_libraries(dummy_manifold_test  io physics  parallel  utilities)

simpla_test(cartesian_test 
    field_vector_calculus_Cartesian_test.cpp 
                       #      field_basic_algerbra_Cartesian_test.cpp 
                             )
target_link_libraries(cartesian_test  io physics  parallel  utilities)

simpla_test(cartesian_complex_test   field_vector_calculus_Cartesian_complex_test.cpp)                              
target_link_libraries(cartesian_complex_test  io physics  parallel  utilities)

simpla_test(cartesian_vec_test   field_vector_calculus_Cartesian_vec3_test.cpp) 
target_link_libraries(cartesian_vec_test  io physics  parallel  utilities)

simpla_test(cartesian_basic_test   field_basic_algerbra_Cartesian_test.cpp )
target_link_libraries(cartesian_basic_test  io physics  parallel  utilities)