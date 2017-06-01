
FILE(GLOB mesh_SRC ./*.cpp)
add_library(mesh ${mesh_SRC} RectMesh.cpp RectMesh.h CoRectMesh.cpp)

TARGET_LINK_LIBRARIES(mesh  toolbox  )

