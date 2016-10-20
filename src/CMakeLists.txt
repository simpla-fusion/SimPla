ADD_SUBDIRECTORY(simpla)

add_library(simpla_lib dummy.cpp)

TARGET_LINK_LIBRARIES(simpla_lib toolbox mesh physics simulation)

add_library(sp_capi SHARED sp_capi.cpp)

TARGET_LINK_LIBRARIES(sp_capi toolbox)