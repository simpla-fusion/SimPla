ADD_SUBDIRECTORY(gtl)

ADD_SUBDIRECTORY(parallel)

ADD_SUBDIRECTORY(io)

ADD_SUBDIRECTORY(task_flow)

ADD_SUBDIRECTORY(dataset)

ADD_SUBDIRECTORY(physics)

ADD_SUBDIRECTORY(manifold)

ADD_SUBDIRECTORY(geometry)

ADD_SUBDIRECTORY(model)

ADD_SUBDIRECTORY(field)

ADD_SUBDIRECTORY(particle)


add_library(simpla_lib)
TARGET_LINK_LIBRARIES(simpla_lib task_flow io parallel physics geometry
        dataset ${LUA_LIBRARIES} ${HDF5_C_LIBRARIES} ${MPI_C_LIBRARIES} utilities)

