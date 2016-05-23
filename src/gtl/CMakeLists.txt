ADD_SUBDIRECTORY( test)
add_library(utilities Log.cpp parse_command_line.cpp ConfigParser.cpp MemoryPool.cpp logo.cpp LuaObject.cpp Properties.cpp )
target_link_libraries(utilities ${LUA_LIBRARIES})

