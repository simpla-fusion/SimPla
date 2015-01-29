
my_test(field_simple_mesh_test     field_simple_mesh_test.cpp        )
target_link_libraries(field_simple_mesh_test   pthread   utilities  )

my_test(field_simple_mesh_map_test     field_simple_mesh_map_test.cpp        )
target_link_libraries(field_simple_mesh_map_test   pthread   utilities  )
 

simpla_test(dummy_manifold_test    field_basic_algerbra_test.cpp)
target_link_libraries(dummy_manifold_test  io physics  parallel  utilities)

simpla_test(cartesian_test 					  		
   							field_vector_calculus_Cartesian_test.cpp 
                            field_basic_algerbra_Cartesian_test.cpp 
                             )
target_link_libraries(cartesian_test  io physics  parallel  utilities)

simpla_test(cartesian_complex_test  
					field_vector_calculus_Cartesian_complex_test.cpp 
                              )
target_link_libraries(cartesian_complex_test  io physics  parallel  utilities)

simpla_test(cartesian_vec3_test  
					field_vector_calculus_Cartesian_vec3_test.cpp 
                              )
target_link_libraries(cartesian_vec3_test  io physics  parallel  utilities)


simpla_test(cylindrical_test 					  		
   							field_vector_calculus_cylindrical_test.cpp 
                            field_basic_algerbra_cylindrical_test.cpp 
                             )
target_link_libraries(cylindrical_test  io physics  parallel  utilities)
