my_test(ntuple_test    )  
target_link_libraries(ntuple_test   parallel utilities )
 

my_test(properties_test    )  
target_link_libraries(properties_test utilities   parallel   physics  utilities)

 
 