add_executable(samrai_test SAMRAI_test.cpp)
target_link_libraries(samrai_test samrai_wrapper mesh toolbox)
