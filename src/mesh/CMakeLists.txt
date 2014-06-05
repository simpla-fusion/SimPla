my_test(mesh_test  mesh_test2.cpp   )  
target_link_libraries(mesh_test  ${LUA_LIBRARIES}  io  physics  ${MPI_LIBRARIES} )
