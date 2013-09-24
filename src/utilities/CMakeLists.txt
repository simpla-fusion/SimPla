my_test(lua_parser_test utilities)
add_library(utilities   properties)
TARGET_LINK_LIBRARIES(utilities ${LUA_LIBRARIES}  ${Boost_LIBRARIES})
