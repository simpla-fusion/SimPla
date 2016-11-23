
FILE(GLOB mesh_SRC ./*.cpp)
add_library(mesh ${mesh_SRC})

TARGET_LINK_LIBRARIES(mesh concept toolbox data)

