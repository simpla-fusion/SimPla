#FILE(GLOB parallel_SRC./*.cpp)
SET(parallel_SRC MPIAuxFunctions.cpp MPIComm.cpp Parallel.cpp)
add_library(parallel ${parallel_SRC})
target_link_libraries(parallel ${MPI_LIBRARIES})