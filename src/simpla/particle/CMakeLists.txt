FILE(GLOB_RECURSE particle_SRC  *.cpp *.h )
add_library(particle ${particle_SRC} )
target_link_libraries(particle utilities ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
set_target_properties(particle PROPERTIES COMPILE_FLAGS "  -xcuda -fno-limit-debug-info -DDEBUG_INITIALIZE_UNDEFINED")
