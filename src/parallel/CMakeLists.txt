
FILE(GLOB PARALLEL_SRC ./*.cpp)
add_library(parallel ${PARALLEL_SRC})
target_link_libraries(parallel ${TBB_LIBRARIES} ${MPI_C_LIBRARIES})

ADD_SUBDIRECTORY(test)
