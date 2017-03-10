FILE(GLOB data_SRC ./*.cpp)
#FILE(GLOB data_SRC DataTable.cpp )
add_library(data ${data_SRC})
add_subdirectory(backend)
