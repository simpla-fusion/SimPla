FILE(GLOB data_SRC ./*.cpp ./backend/*.cpp)
#FILE(GLOB data_SRC DataTable.cpp )
add_library(data ${data_SRC})

