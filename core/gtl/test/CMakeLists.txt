my_test(ntuple_test   
   ntuple_test.cpp 
   ntuple_perf_test.cpp 
   ntuple_reduce_test.cpp 
    )  
target_link_libraries(ntuple_test     utilities )
 

my_test(properties_test    )  
target_link_libraries(properties_test utilities   parallel   physics  utilities)

 
 