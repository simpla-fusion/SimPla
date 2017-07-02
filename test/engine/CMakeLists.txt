simpla_test(engine_test engine_test.cpp)
target_link_libraries(engine_test engine data utilities)

simpla_test(AttributeView_test   AttributeView_test.cpp)
target_link_libraries(AttributeView_test engine  data utilities)