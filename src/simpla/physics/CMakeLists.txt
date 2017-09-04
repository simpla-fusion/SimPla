FILE(GLOB_RECURSE physics_SRC  *.cpp *.h )
add_library(physics ${physics_SRC} )
target_link_libraries(physics utilities ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
set_target_properties(physics PROPERTIES COMPILE_FLAGS "  -xcuda -fno-limit-debug-info -DDEBUG_INITIALIZE_UNDEFINED")
