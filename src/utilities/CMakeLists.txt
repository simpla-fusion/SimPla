ADD_EXECUTABLE(lua_state_test lua_state_test.cpp)
TARGET_LINK_LIBRARIES(lua_state_test   ${LUA_LIBRARIES}  )

ADD_EXECUTABLE(geqdsk_test geqdsk.cpp geqdsk_test.cpp) 
TARGET_LINK_LIBRARIES(geqdsk_test   Xdmf)