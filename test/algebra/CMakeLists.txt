
simpla_test(field_basic_algebra_test field_basic_algebra_test.cpp)
target_link_libraries(field_basic_algebra_test  mesh utilities)
#
simpla_test(field_diff_calculus_cartesian_test field_diff_calculus_test.cpp)
#target_compile_options(field_diff_calculus_cartesian_test PUBLIC -DCARTESIAN_COORDINATE_SYSTEM)
target_link_libraries(field_diff_calculus_cartesian_test  mesh utilities)

simpla_test(field_aritmetic_test  field_aritmetic_test.cpp)


add_executable(field_dummy field_dummy.cpp)
target_link_libraries(field_dummy
        netcdf ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES} ${TBB_LIBRARIES}
        -Wl,--whole-archive utilities geometry  data  parallel mesh engine model
            -Wl,--no-whole-archive   ${OPENMP_LIBRARIES} ${CUDA_LIBRARIES}
        )





