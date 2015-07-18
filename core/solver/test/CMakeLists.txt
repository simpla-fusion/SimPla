simpla_test(field_diff_calculus_cartesian_test 	 field_diff_calculus_test.cpp    )
target_compile_options(field_diff_calculus_cartesian_test PUBLIC   -DCARTESIAN_COORDINATE_SYTEM)
target_link_libraries(field_diff_calculus_cartesian_test  simpla_aux)

simpla_test(field_diff_calculus_cylindrical_test 	 field_diff_calculus_test.cpp    )
target_compile_options(field_diff_calculus_cylindrical_test PUBLIC   -CYLINDRICAL_COORDINATE_SYTEM )
target_link_libraries(field_diff_calculus_cylindrical_test  simpla_aux)

simpla_test(field_basic_algerbra_test 	 field_basic_algerbra_test.cpp    )
target_link_libraries(field_basic_algerbra_test  simpla_aux)

simpla_test(mesh_test 	 mesh_test.cpp    )
target_link_libraries(mesh_test  simpla_aux)