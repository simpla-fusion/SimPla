#my_test(field_simple_mesh_shared_ptr_test     field_simple_mesh_shared_ptr_test.cpp        )
#target_link_libraries(field_simple_mesh_shared_ptr_test   pthread   utilities  )
#
#my_test(field_simple_mesh_map_test     field_simple_mesh_map_test.cpp        )
#target_link_libraries(field_simple_mesh_map_test   pthread   utilities  )
#
#my_test(field_cartesian_test    field_cartesian_map_test.cpp)
#target_link_libraries(field_cartesian_map_test  pthread physics utilities)
#
#simpla_test(cartesian_test
#   							field_vector_calculus_Cartesian_test.cpp
#        field_basic_algebra_Cartesian_test.cpp
#                             )
#target_link_libraries(cartesian_test  io physics  parallel  utilities)
#
#simpla_test(cartesian_complex_test
#					field_vector_calculus_Cartesian_complex_test.cpp
#                              )
#target_link_libraries(cartesian_complex_test  io physics  parallel  utilities)
#
#simpla_test(cartesian_vec3_test
#					field_vector_calculus_Cartesian_vec3_test.cpp
#                              )
#target_link_libraries(cartesian_vec3_test  io physics  parallel  utilities)
#
#
#simpla_test(cylindrical_test
#   							field_vector_calculus_cylindrical_test.cpp
#        field_basic_algebra_cylindrical_test.cpp
#                             )
#target_link_libraries(cylindrical_test  io physics  parallel  utilities)
simpla_test(field_basic_algebra_test field_basic_algebra_test.cpp)
target_link_libraries(field_basic_algebra_test simpla_lib)

simpla_test(field_diff_calculus_cartesian_test field_diff_calculus_test.cpp)
target_compile_options(field_diff_calculus_cartesian_test PUBLIC -DCARTESIAN_COORDINATE_SYSTEM)
target_link_libraries(field_diff_calculus_cartesian_test simpla_lib)

#simpla_test(field_diff_calculus_cylindrical_test field_diff_calculus_test.cpp)
#target_compile_options(field_diff_calculus_cylindrical_test PUBLIC -DCYLINDRICAL_COORDINATE_SYSTEM)
#target_link_libraries(field_diff_calculus_cylindrical_test simpla_lib)


add_executable(field_dummy field_dummy.cpp)
target_link_libraries(field_dummy simpla_lib)
