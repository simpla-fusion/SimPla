ADD_SUBDIRECTORY(base)

ADD_SUBDIRECTORY(toolbox)

ADD_SUBDIRECTORY(physics)

ADD_SUBDIRECTORY(model)

ADD_SUBDIRECTORY(mesh)

ADD_SUBDIRECTORY(manifold)

ADD_SUBDIRECTORY(simulation)

add_library(simpla_lib dummy.cpp)

TARGET_LINK_LIBRARIES(simpla_lib base toolbox mesh physics simulation)

add_library(sp_capi SHARED sp_capi.cpp)

TARGET_LINK_LIBRARIES(sp_capi base toolbox)