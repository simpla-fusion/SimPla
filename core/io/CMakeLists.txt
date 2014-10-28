
add_library(io  data_stream.cpp  )
TARGET_LINK_LIBRARIES(io utilities    ${HDF5_LIBRARIES})


my_test(data_stream_test    )
target_link_libraries(data_stream_test   io parallel utilities  )
