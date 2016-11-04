FILE(GLOB simulation_SRC ./*.cpp) #${simulation_SRC}
add_library(simulation Worker.cpp)

add_subdirectory(main)