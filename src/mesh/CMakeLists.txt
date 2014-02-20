my_test(mesh_test   )  
target_link_libraries(mesh_test ${LUA_LIBRARIES}  io  physics )
add_library(mesh topology_rect.cpp co_rect_mesh_rz.cpp)
TARGET_LINK_LIBRARIES(mesh physics  ${LUA_LIBRARIES} io)