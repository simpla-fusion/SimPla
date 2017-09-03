simpla_test(DataTable_test DataNode_test.cpp)
target_link_libraries(DataTable_test
        -Wl,--whole-archive
          data utilities data_backend
        -Wl,--no-whole-archive

        )
