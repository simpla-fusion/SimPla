FILE(GLOB simulation_SRC ./*.cpp) #
add_library(simulation ${simulation_SRC})
TARGET_LINK_LIBRARIES(simulation data)
add_subdirectory(main)