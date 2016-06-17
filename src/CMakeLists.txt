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

 
add_library(simpla_lib dummy.cpp)
TARGET_LINK_LIBRARIES(simpla_lib
        ${MPI_C_LIBRARIES} ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES} ${TBB_LIBRARIES} ${LUA_LIBRARIES}
        base
        task_flow
        mesh
        data_model io
        parallel
        physics
        #        geometry
        utilities
        particle
        )

