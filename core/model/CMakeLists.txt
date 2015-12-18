

ADD_EXECUTABLE(geqdsk_convert GEqdsk.cpp)
TARGET_LINK_LIBRARIES(geqdsk_convert simpla_lib)


add_library(geqdsk GEqdsk.cpp)
TARGET_LINK_LIBRARIES(geqdsk simpla_lib)
