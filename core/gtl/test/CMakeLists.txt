#simpla_test(ntuple_test
#   ntuple_test.cpp
#   ntuple_perf_test.cpp
#   ntuple_reduce_test.cpp
#    )
#target_link_libraries(ntuple_test     utilities )


#simpla_test(properties_test    )
#target_link_libraries(properties_test utilities   parallel   physics  utilities)


simpla_test(log_test ../utilities/log.cpp ../utilities/parse_command_line.cpp)
#target_link_libraries(log_test      )

#
#simpla_test(memory_pool_test   )
#target_link_libraries(memory_pool_test utilities io parallel )
#
simpla_test(lua_state_test lua_state_test.cpp)
target_link_libraries(lua_state_test utilities data_model base    )
