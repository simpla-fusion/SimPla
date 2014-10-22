ADD_EXECUTABLE(lua_state_test lua_state_test.cpp)
TARGET_LINK_LIBRARIES(lua_state_test       parallel   physics  utilities  )


add_library(utilities   properties.cpp log.cpp  )

my_test(ntuple_test    )  
target_link_libraries(ntuple_test )

ADD_EXECUTABLE(ntuple_test2 ntuple_test2.cpp)
TARGET_LINK_LIBRARIES(ntuple_test2    )


my_test(properties_test    )  
target_link_libraries(properties_test utilities   parallel   physics  utilities)
my_test(log_test  log.cpp    )  
target_link_libraries(log_test   ${MPI_LIBRARIES})