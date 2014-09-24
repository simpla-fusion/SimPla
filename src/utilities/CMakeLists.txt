ADD_EXECUTABLE(lua_state_test lua_state_test.cpp)
TARGET_LINK_LIBRARIES(lua_state_test       parallel   physics  utilities  )


add_library(utilities   properties.cpp log.cpp  )

my_test(ntuple_test    )  
target_link_libraries(ntuple_test ${EXT_LIBRARIES}  utilities)

my_test(properties_test    )  
target_link_libraries(properties_test utilities   parallel   physics  utilities)
my_test(log_test  log.cpp    )  
target_link_libraries(log_test   ${MPI_LIBRARIES})