ADD_SUBDIRECTORY(test)
add_library(utilities Log.cpp MemoryPool.cpp logo.cpp Properties.cpp parse_command_line.cpp)


###########################################################
#IF (DEFINED ENV{LUA_ROOT})
#
#    set(LUA_ROOT $ENV{LUA_ROOT})
#
#    find_path(LUA_INCLUDE_DIR
#            NAMES lua.h
#            HINTS ${MPI_ROOT}/include/
#            DOC "MPI header file path"
#            )
#    SET(LUA_LIBRARY_DIRS ${LUA_ROOT}/lib/)
#
#    find_library(LUA_LIBRARY liblua.a HINTS ${LUA_LIBRARY_DIRS})
#
#    SET(LUA_LIBRARIES ${LUA_LIBRARY})
#
#    FIND_PACKAGE_HANDLE_STANDARD_ARGS(LUA DEFAULT_MSG LUA_LIBRARIES LUA_INCLUDE_DIRS LUA_ROOT)


FIND_PACKAGE(Lua 5.2 REQUIRED)

IF (LUA_FOUND)
    INCLUDE_DIRECTORIES(${LUA_INCLUDE_DIR})
ENDIF (LUA_FOUND)


add_library(lua_parser LuaObject.cpp ConfigParser.cpp)
target_link_libraries(lua_parser ${LUA_LIBRARIES} )


