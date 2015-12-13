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

add_library(simpla_lib dummy.cpp manifold/mesh/mesh_map.h manifold/mesh/mesh_graph.h manifold/mesh/samr_object.h manifold/enable_patch_from_this.h)
TARGET_LINK_LIBRARIES(simpla_lib
        ${MPI_C_LIBRARIES} ${HDF5_C_LIBRARIES} ${TBB_LIBRARIES} ${LUA_LIBRARIES}
        task_flow mesh_policies dataset io parallel physics geometry utilities
        )

