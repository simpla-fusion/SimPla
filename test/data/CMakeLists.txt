simpla_test(DataTable_test DataTable_test.cpp)
target_link_libraries(DataTable_test
        -Wl,--whole-archive

        data_backend  data utilities
        -Wl,--no-whole-archive

        )
