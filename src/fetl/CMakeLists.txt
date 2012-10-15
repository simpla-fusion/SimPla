my_test(fetl_test utilities )
my_test(ntuple_test  utilities fetl)
#add_executable(ntuple_test   test/ntuple_test.cpp)
#target_link_libraries(ntuple_test gtest  )

add_library(fetl grid/grid.cpp) 	 