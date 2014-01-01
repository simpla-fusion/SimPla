add_library(physics physical_constants.cpp)
target_link_libraries(physics  ${LUA_LIBRARIES} )