FILE(GLOB engine_SRC ./*.cpp)
#SET(engine_SRC AttributeView.cpp DomainView.cpp AttributeDesc.cpp SPObject.cpp Worker.cpp)
add_library(engine ${engine_SRC})