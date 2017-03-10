add_executable(data_dummy data_dummy.cpp )
target_link_libraries(data_dummy  data toolbox data_backend)
simpla_test(DataType_test.cpp DataType_test.cpp)
target_link_libraries(DataType_test.cpp data toolbox)