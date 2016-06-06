add_library(mesh MeshUtility.cpp MeshAttribute.cpp)
target_link_libraries(mesh data_model parallel)