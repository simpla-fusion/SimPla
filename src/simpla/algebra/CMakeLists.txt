FILE(GLOB algebra_SRC  *.cpp *.h )
add_library(algebra ${algebra_SRC})
#target_link_libraries(physics ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${LUA_LIBRARIES} ${TBB_LIBRARIES})

