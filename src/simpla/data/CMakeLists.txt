FILE(GLOB data_SRC ./*.cpp ./backend/*.cpp)
#FILE(GLOB data_SRC DataTable.cpp )
add_library(data ${data_SRC})
target_link_libraries(data ${LUA_LIBRARIES})
