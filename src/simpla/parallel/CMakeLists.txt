FILE(GLOB parallel_SRC ./*.cpp)
add_library(parallel ${parallel_SRC})
target_link_libraries(parallel ${MPI_LIBRARIES})