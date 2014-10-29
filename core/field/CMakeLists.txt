
my_test(field_io_test     field_io_test.cpp        )
target_link_libraries(field_io_test physics io parallel utilities)



my_test(field_basic_algerbra_test )
target_link_libraries(field_basic_algerbra_test   parallel  utilities)

my_test(field_vector_calculus_Cartesian_test )
target_link_libraries(field_vector_calculus_Cartesian_test  io physics    parallel  utilities)