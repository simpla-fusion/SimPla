 
my_test(log_test   ../log.cpp  ../parse_command_line.cpp )  
#target_link_libraries(log_test      )


my_test(memory_pool_test   )  
target_link_libraries(memory_pool_test utilities io parallel )

my_test(lua_state_test  ../log.cpp   ../parse_command_line.cpp)  
target_link_libraries(lua_state_test    ${LUA_LIBRARIES}  )
 