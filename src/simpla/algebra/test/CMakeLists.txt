simpla_test(ntuple_test ntuple_test.cpp
        ntuple_perf_test.cpp
        #  ntuple_reduce_test.cpp
        )
simpla_test(field_basic_algebra_test field_basic_algebra_test.cpp)

simpla_test(field_diff_calculus_cartesian_test field_diff_calculus_test.cpp)
target_compile_options(field_diff_calculus_cartesian_test PUBLIC -DCARTESIAN_COORDINATE_SYSTEM)
 