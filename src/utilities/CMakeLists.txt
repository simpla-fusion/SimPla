ADD_EXECUTABLE(lua_state_test lua_state_test.cpp)
TARGET_LINK_LIBRARIES(lua_state_test   ${LUA_LIBRARIES} ${MPI_LIBRARIES} )

#my_test(log_test  pthread )  
#target_link_libraries(log_test pthread  )