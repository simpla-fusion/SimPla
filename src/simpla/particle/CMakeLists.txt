FILE(GLOB_RECURSE particle_SRC ./*.cpp )
add_library(particle ${particle_SRC} )
target_link_libraries(particle utilities )