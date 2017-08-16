

FILE(GLOB model_SRC  *.cpp *.h  )
add_library(model ${model_SRC})

add_subdirectory(occ)

TARGET_LINK_LIBRARIES(model    utilities data model_occ )

#FILE(GLOB geometry_SRC ./*.cpp)
#add_library(geometry ${geometry_SRC} Cube.cpp CoordinateSystem.cpp)
