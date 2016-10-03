ADD_SUBDIRECTORY(base)

ADD_SUBDIRECTORY(toolbox)

ADD_SUBDIRECTORY(parallel)

ADD_SUBDIRECTORY(io)

ADD_SUBDIRECTORY(data_model)

ADD_SUBDIRECTORY(physics)

ADD_SUBDIRECTORY(model)

ADD_SUBDIRECTORY(mesh)

ADD_SUBDIRECTORY(manifold)

ADD_SUBDIRECTORY(field)

ADD_SUBDIRECTORY(particle)

ADD_SUBDIRECTORY(simulation)

add_library(simpla_lib dummy.cpp)

TARGET_LINK_LIBRARIES(simpla_lib
        base data_model io parallel
        utilities lua_parser
        mesh physics
        simulation
        )

add_library(sp_capi SHARED sp_capi.cpp)
TARGET_LINK_LIBRARIES(sp_capi base data_model io parallel utilities)