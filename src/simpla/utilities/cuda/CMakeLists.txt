FILE(GLOB toolbox_cuda_SRC ./*.cpp)
add_library(toolbox_cuda ${toolbox_cuda_SRC}  )
set_target_properties(toolbox_cuda PROPERTIES COMPILE_FLAGS "  -xcuda")
target_link_libraries(toolbox_cuda  ${CUDA_LIBRARIES})