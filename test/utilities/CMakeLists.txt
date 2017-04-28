simpla_test(ntuple_test ntuple_test.cpp
        #  ntuple_reduce_test.cpp
        )


add_executable(ntuple_dummy ntuple_dummy.cpp)



add_executable(ntuple_bench  ntuple_bench.cpp)
target_link_libraries(ntuple_bench benchmark pthread)

