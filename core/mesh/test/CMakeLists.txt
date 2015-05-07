simpla_test(field_diff_calculus_test 	 field_diff_calculus_test.cpp    )
target_link_libraries(field_diff_calculus_test  simpla_aux)

simpla_test(field_basic_algerbra_test 	 field_basic_algerbra_test.cpp    )
target_link_libraries(field_basic_algerbra_test  simpla_aux)

simpla_test(mesh_test 	 mesh_test.cpp    )
target_link_libraries(mesh_test  simpla_aux)