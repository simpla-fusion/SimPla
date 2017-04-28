simpla_test(ntuple_test ntuple_test.cpp )


add_executable(ntuple_dummy ntuple_dummy.cpp)


add_executable(ntuple_bench ntuple_bench.cpp)
target_link_libraries(ntuple_bench benchmark pthread)

add_executable(array_dummy array_dummy.cpp)
target_link_libraries(array_dummy data)