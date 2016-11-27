FILE(GLOB model_SRC ./*.cpp)
add_library(model ${model_SRC})

TARGET_LINK_LIBRARIES(model concept toolbox data)

