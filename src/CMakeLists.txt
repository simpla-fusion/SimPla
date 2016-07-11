ADD_SUBDIRECTORY(base)

ADD_SUBDIRECTORY(gtl)

ADD_SUBDIRECTORY(parallel)

ADD_SUBDIRECTORY(io)

ADD_SUBDIRECTORY(task_flow)

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
        base
        task_flow
        mesh
        data_model io
        parallel
        physics
        #        geometry
        utilities
        lua_parser
        simulation
        )

add_library(sp_cwrap sp_cwrap.cpp)
TARGET_LINK_LIBRARIES(sp_cwrap base data_model io parallel utilities


        # base
        # mesh
        # physics
        # geometry
        # utilities
        # simulation
        )