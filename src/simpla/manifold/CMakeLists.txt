
FILE(GLOB manifold_SRC ./*.cpp)
add_library(manifold ${manifold_SRC})

TARGET_LINK_LIBRARIES(manifold concept toolbox data)

