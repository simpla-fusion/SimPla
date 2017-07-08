FILE(GLOB engine_SRC ./*.cpp)
add_library(engine ${engine_SRC})
target_link_libraries(engine model data parallel)
