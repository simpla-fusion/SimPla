FILE(GLOB model_SRC  *.cpp  )
add_library(model ${model_SRC}  )
TARGET_LINK_LIBRARIES(model    utilities data ${OCE_LIBRARIES})

#FILE(GLOB geometry_SRC ./*.cpp)
#add_library(geometry ${geometry_SRC} Cube.cpp CoordinateSystem.cpp)
