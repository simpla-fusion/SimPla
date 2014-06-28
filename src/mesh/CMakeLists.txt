my_test(topology_UniformArray_test    )  
target_link_libraries(topology_UniformArray_test  ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )


my_test(geometry_cartesian_test     )  
target_link_libraries(geometry_cartesian_test   ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )

my_test(geometry_cylindrical_test    )  
target_link_libraries(geometry_cylindrical_test  ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )


my_test(mesh_test    )  
target_link_libraries(mesh_test  ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )
