add_library(mesh MeshBlock.cpp MeshBlockStructured.cpp MeshBlockStructured.h MeshEntitySet.h)
target_link_libraries(mesh data_model)