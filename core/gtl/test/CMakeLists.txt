my_test(ntuple_test
   ntuple_test.cpp
   ntuple_perf_test.cpp
   ntuple_reduce_test.cpp
    )  
target_link_libraries(ntuple_test     utilities )
 

my_test(properties_test    )  
target_link_libraries(properties_test utilities   parallel   physics  utilities)

 

my_test(log_test   ../log.cpp  ../parse_command_line.cpp )
#target_link_libraries(log_test      )


my_test(memory_pool_test   )
target_link_libraries(memory_pool_test utilities io parallel )

my_test(lua_state_test  ../log.cpp   ../parse_command_line.cpp)
target_link_libraries(lua_state_test    ${LUA_LIBRARIES}  )
