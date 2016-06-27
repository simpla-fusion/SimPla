FILE(GLOB_RECURSE mesh_SRC ./*.cpp)
add_library(mesh ${mesh_SRC})
target_link_libraries(mesh data_model parallel)