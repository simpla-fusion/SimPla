

ADD_EXECUTABLE(geqdsk_convert geqdsk.cpp geqdsk_convert.cpp)
TARGET_LINK_LIBRARIES(geqdsk_convert   Xdmf simpla_lib )


add_library(geqdsk   geqdsk.cpp)
TARGET_LINK_LIBRARIES(geqdsk   simpla_lib )
