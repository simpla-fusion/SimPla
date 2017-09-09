simpla_test(Attribute_test Attribute_test.cpp)
target_link_libraries(Attribute_test
        -Wl,--whole-archive
      algebra engine geometry data utilities data_backend
        -Wl,--no-whole-archive

        )

ADD_EXECUTABLE(Attribute_dummy  Attribute_dummy.cpp)
target_link_libraries(Attribute_dummy
        -Wl,--whole-archive
        algebra engine geometry data utilities data_backend
        -Wl,--no-whole-archive

        )