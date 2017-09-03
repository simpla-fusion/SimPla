#
simpla_test(GeoObject_test GeoObject_test.cpp ../../src/simpla/geometry/GeoObject.cpp)
target_link_libraries(GeoObject_test
        -Wl,--whole-archive
        data data_backend
        -Wl,--no-whole-archive

        )
#
