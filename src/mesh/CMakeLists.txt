add_library(mesh MeshAtlas.cpp AMR.cpp)
target_link_libraries(mesh data_model)