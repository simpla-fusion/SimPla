add_library(mesh MeshBlock.cpp obsolete/MeshBlockStructured.cpp MeshAttribute.cpp)
target_link_libraries(mesh data_model)