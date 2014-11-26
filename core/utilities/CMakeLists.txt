
add_library(utilities   properties.cpp log.cpp parse_command_line.cpp config_parser.cpp  )
TARGET_LINK_LIBRARIES(utilities )

 
my_test(ntuple_test    )  
target_link_libraries(ntuple_test )
 

my_test(properties_test    )  
target_link_libraries(properties_test utilities   parallel   physics  utilities)

my_test(log_test  log.cpp    )  
target_link_libraries(log_test   parallel)


my_test(memory_pool_test   )  
target_link_libraries(memory_pool_test utilities io parallel )
 