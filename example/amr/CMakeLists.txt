FILE(GLOB amr_test_SRC ./*.cpp)
add_executable(amr_test ${amr_test_SRC})
target_link_libraries(amr_test mesh toolbox simulation)