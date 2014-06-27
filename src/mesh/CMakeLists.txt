my_test(topology_test_UniformArray    )  
target_link_libraries(topology_test_UniformArray  ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )


my_test(mesh_test_Cartesian    )  
target_link_libraries(mesh_test_Cartesian  ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )
my_test(mesh_test_Cylindrical    )  
target_link_libraries(mesh_test_Cylindrical  ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )
