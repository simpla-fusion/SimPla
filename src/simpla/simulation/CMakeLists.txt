FILE(GLOB simulation_SRC ./*.cpp) #${simulation_SRC}
add_library(simulation ../mesh/Worker.cpp)

add_subdirectory(main)