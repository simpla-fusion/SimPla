FILE(GLOB data_SRC ./*.cpp)
#FILE(GLOB data_SRC DataTable.cpp DataEntity.cpp DataType.cpp HeavyData.cpp LightData.cpp DataTableAdapterLua.cpp)
add_library(data ${data_SRC})