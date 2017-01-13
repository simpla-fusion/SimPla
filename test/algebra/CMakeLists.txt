simpla_test(ntuple_test ntuple_test.cpp
         #  ntuple_reduce_test.cpp
        )
simpla_test(field_basic_algebra_test field_basic_algebra_test.cpp)
target_link_libraries(field_basic_algebra_test  mesh toolbox)
#
simpla_test(field_diff_calculus_cartesian_test field_diff_calculus_test.cpp)
#target_compile_options(field_diff_calculus_cartesian_test PUBLIC -DCARTESIAN_COORDINATE_SYSTEM)
target_link_libraries(field_diff_calculus_cartesian_test  mesh toolbox)

simpla_test(field_aritmetic_test  field_aritmetic_test.cpp)


add_executable(field_dummy field_dummy.cpp)
target_link_libraries(field_dummy   mesh data)

add_executable(ntuple_dummy ntuple_dummy.cpp)


add_executable(array_dummy array_dummy.cpp)
target_link_libraries(array_dummy data)

add_executable(ntuple_bench ntuple_bench.cpp)
target_link_libraries(ntuple_bench benchmark pthread)

