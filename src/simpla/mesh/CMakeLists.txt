
add_subdirectory(wrapper)
FILE(GLOB mesh_SRC ./*.cpp)
add_library(mesh ${mesh_SRC} wrapper)


