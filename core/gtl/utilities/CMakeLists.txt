add_library(utilities log.cpp parse_command_line.cpp config_parser.cpp memory_pool.cpp logo.cpp)
target_link_libraries(utilities ${LUA_LIBRARIES})
