FILE(GLOB data_SRC DataTable.cpp DataEntity.cpp DataType.cpp HeavyData.cpp LightData.cpp)
add_library(data ${data_SRC})