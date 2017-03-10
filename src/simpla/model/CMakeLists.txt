FILE(GLOB model_SRC ./*.cpp ./geometry/*.cpp)
add_library(model ${model_SRC})

TARGET_LINK_LIBRARIES(model   toolbox data )

