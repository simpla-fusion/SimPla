FILE(GLOB physics_SRC  *.cpp *.h )
add_library(physics ${physics_SRC} )
target_link_libraries(physics utilities )
