
add_library(io  io.cpp data_stream.cpp  )
TARGET_LINK_LIBRARIES(io  ${HDF5_LIBRARIES} dataset)


my_test(data_stream_test    )
target_link_libraries(data_stream_test   io parallel utilities  dataset)
