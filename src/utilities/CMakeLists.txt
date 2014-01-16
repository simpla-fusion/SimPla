ADD_EXECUTABLE(lua_state_test lua_state_test.cpp)
TARGET_LINK_LIBRARIES(lua_state_test   ${LUA_LIBRARIES}  )

ADD_EXECUTABLE(geqdsk_test geqdsk.cpp geqdsk_test.cpp) 
TARGET_LINK_LIBRARIES(geqdsk_test   Xdmf)


add_library(utilities   geqdsk.cpp)
TARGET_LINK_LIBRARIES(utilities ${LUA_LIBRARIES} Xdmf )

#my_test(log_test  pthread )  
#target_link_libraries(log_test pthread  )