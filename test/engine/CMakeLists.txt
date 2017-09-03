simpla_test(SPObject_test SPObject_test.cpp ../../src/simpla/engine/SPObject.cpp     )

target_link_libraries(SPObject_test
        -Wl,--whole-archive
        data data_backend
        -Wl,--no-whole-archive

        )