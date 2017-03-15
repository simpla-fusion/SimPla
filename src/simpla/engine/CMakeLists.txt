FILE(GLOB engine_SRC ./*.cpp)
add_library(engine ${engine_SRC})
add_subdirectory(main)