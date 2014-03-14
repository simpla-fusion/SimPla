my_test(mesh_rectangle_test   )  
target_link_libraries(mesh_rectangle_test ${LUA_LIBRARIES}  io  physics )
add_library(mesh   )
TARGET_LINK_LIBRARIES(mesh physics  ${LUA_LIBRARIES} io)