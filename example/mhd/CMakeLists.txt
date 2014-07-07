ADD_EXECUTABLE(mhd  mhd.cpp   )
TARGET_LINK_LIBRARIES(mhd  physics  utilities io ${MPI_LIBRARIES})

