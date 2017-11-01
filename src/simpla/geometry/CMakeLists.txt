FILE(GLOB geometry_SRC *.cpp *.h)
add_library(geometry ${geometry_SRC})
add_subdirectory(backend)

TARGET_LINK_LIBRARIES(geometry utilities data geometry_backend)

#FILE(GLOB geometry_SRC ./*.cpp)
#add_library(geometry ${geometry_SRC} Cube.cpp CoordinateSystem.cpp)
