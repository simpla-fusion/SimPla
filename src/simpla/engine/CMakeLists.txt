FILE(GLOB engine_SRC ./*.cpp)
#SET(engine_SRC AttributeView.cpp DomainView.cpp AttributeDesc.cpp Object.cpp Worker.cpp)
add_library(engine ${engine_SRC})