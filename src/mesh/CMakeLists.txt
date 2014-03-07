my_test(rect_mesh_test   )  
target_link_libraries(rect_mesh_test ${LUA_LIBRARIES}  io  physics )
add_library(mesh   )
TARGET_LINK_LIBRARIES(mesh physics  ${LUA_LIBRARIES} io)