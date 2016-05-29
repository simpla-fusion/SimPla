SET(DATASET_SRC DataSpace.cpp DataType.cpp DataSet.cpp)
add_library(data_model ${DATASET_SRC} )
TARGET_LINK_LIBRARIES(data_model base)