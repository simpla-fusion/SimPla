FILE(GLOB engine_SRC  *.cpp *.h )
add_library(engine ${engine_SRC})
target_link_libraries(engine model data parallel)
