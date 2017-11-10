#FILE(GLOB parallel_SRC./*.cpp)


SET(parallel_SRC MPIComm.h MPIComm.cpp Parallel.cpp MPIUpdater.cpp MPIUpdater.h)
add_library(parallel ${parallel_SRC})
target_link_libraries(parallel ${MPI_C_LIBRARIES})
target_include_directories(parallel BEFORE PRIVATE ${MPI_C_INCLUDE_PATH})
