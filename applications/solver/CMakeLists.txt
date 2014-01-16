add_library(field_solver  electromagnetic/cold_fluid.cpp electromagnetic/pml.cpp)
TARGET_LINK_LIBRARIES(field_solver physics  utilities io)