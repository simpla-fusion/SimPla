add_library(mesh MeshAttribute.cpp MeshWorker.cpp)
target_link_libraries(mesh data_model)