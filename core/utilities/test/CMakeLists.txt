my_test(ntuple_test    )  
target_link_libraries(ntuple_test )
 

my_test(properties_test    )  
target_link_libraries(properties_test utilities   parallel   physics  utilities)

my_test(log_test      )  
target_link_libraries(log_test   parallel utilities)


my_test(memory_pool_test   )  
target_link_libraries(memory_pool_test utilities io parallel )

my_test(lua_state_test   )  
target_link_libraries(lua_state_test    lua5.2 utilities)
 