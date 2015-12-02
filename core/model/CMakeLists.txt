

ADD_EXECUTABLE(geqdsk_convert geqdsk.cpp)
TARGET_LINK_LIBRARIES(geqdsk_convert simpla_lib)


add_library(geqdsk geqdsk.cpp)
TARGET_LINK_LIBRARIES(geqdsk simpla_lib)
