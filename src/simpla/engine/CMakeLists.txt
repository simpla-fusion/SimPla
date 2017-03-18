FILE(GLOB engine_SRC ./*.cpp)
add_library(engine ${engine_SRC})
target_link_libraries(engine geometry data)
add_subdirectory(main)