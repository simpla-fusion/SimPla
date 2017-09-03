simpla_test(DataNode_test DataNode_test.cpp)
target_link_libraries(DataNode_test
        -Wl,--whole-archive
          data utilities data_backend
        -Wl,--no-whole-archive

        )
simpla_test(SPObject_test SPObject_test.cpp    )

target_link_libraries(SPObject_test
        -Wl,--whole-archive
        data data_backend
        -Wl,--no-whole-archive

        )