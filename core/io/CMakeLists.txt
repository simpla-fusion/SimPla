SET(IO_SRC io.cpp data_stream.cpp)
add_library(io ${IO_SRC} ${MPI_C_LIBRARIES} ${HDF5_C_LIBRARIES})


#my_test(data_stream_test    )
#target_link_libraries(data_stream_test   io parallel utilities  dataset)
