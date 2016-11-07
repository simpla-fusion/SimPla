FILE(GLOB simulation_SRC ./*.cpp) #
add_library(simulation ${simulation_SRC})

add_subdirectory(main)