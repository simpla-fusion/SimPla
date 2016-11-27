add_subdirectory(test)

FILE(GLOB geometry_SRC ./*.cpp)
add_library(geometry ${geometry_SRC})

