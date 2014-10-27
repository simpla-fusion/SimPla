SET(HDF5_USE_STATIC_LIBRARIES TRUE)
FIND_PACKAGE(HDF5 1.8 COMPONENTS CXX HL )
include_directories(BEFORE ${HDF5_INCLUDE_DIR} )

add_library(io  data_stream.cpp  )

TARGET_LINK_LIBRARIES(io utilities ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES} hdf5 hdf5_hl)


my_test(data_stream_test           )
target_link_libraries(data_stream_test   io parallel utilities ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES})
