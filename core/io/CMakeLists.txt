SET(IO_SRC io.cpp data_stream.cpp xdmf_stream.cpp)

link_directories(${HDF5_LIBRARY_DIRS})
add_library(io ${IO_SRC})
target_link_libraries(io ${HDF5_C_LIBRARIES} dataset)

#my_test(data_stream_test    )
#target_link_libraries(data_stream_test   io parallel utilities  dataset)
