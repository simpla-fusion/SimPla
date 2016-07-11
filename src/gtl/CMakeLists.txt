ADD_SUBDIRECTORY(test)
add_library(utilities Log.cpp MemoryPool.cpp logo.cpp Properties.cpp parse_command_line.cpp)

add_library(lua_parser LuaObject.cpp ConfigParser.cpp)


