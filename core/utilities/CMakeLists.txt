FIND_PACKAGE(Lua52)
INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})

add_library(utilities   log.cpp parse_command_line.cpp config_parser.cpp memory_pool.cpp )
TARGET_LINK_LIBRARIES(utilities ${LUA_LIBRARIES}   )
 
ADD_SUBDIRECTORY( test)
 

 