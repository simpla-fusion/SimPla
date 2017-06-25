
add_subdirectory(cuda)
#target_link_libraries(toolbox ${LUA_LIBRARIES} )

FILE(GLOB toolbox_SRC ./*.cpp ./*.cu)
add_library(toolbox ${toolbox_SRC} )
MESSAGE(${toolbox_SRC})
target_link_libraries(toolbox ${LUA_LIBRARIES}  toolbox_cuda)

#${HDF5_LIBRARIES} ${MPI_LIBRARIES}  ${TBB_LIBRARIES}
###########################################################
#IF (DEFINED ENV{LUA_ROOT})
#
#    SetValue(LUA_ROOT $ENV{LUA_ROOT})
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

#
#add_library(lua_parser LuaObject.cpp ConfigParser.cpp)
#target_link_libraries(lua_parser ${LUA_LIBRARIES})


