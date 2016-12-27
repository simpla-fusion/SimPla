FILE(GLOB physics_SRC ./*.cpp)
add_library(physics ${physics_SRC})
#target_link_libraries(physics ${HDF5_LIBRARIES} ${MPI_LIBRARIES} ${LUA_LIBRARIES} ${TBB_LIBRARIES})

