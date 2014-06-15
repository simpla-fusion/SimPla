ADD_EXECUTABLE(lua_state_test lua_state_test.cpp)
TARGET_LINK_LIBRARIES(lua_state_test   ${EXT_LIBRARIES}  )

my_test(ntuple_test    )  
target_link_libraries(ntuple_test ${EXT_LIBRARIES}  )