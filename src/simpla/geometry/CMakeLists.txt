
FILE(GLOB geometry_SRC ./*.cpp)
add_library(geometry ${geometry_SRC} Cube.cpp CoordinateSystem.cpp)

