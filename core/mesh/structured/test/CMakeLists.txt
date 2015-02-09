my_test(geometry_cartesian_test     )  
target_link_libraries(geometry_cartesian_test   parallel   physics  utilities dataset )

my_test(geometry_cylindrical_test    )  
target_link_libraries(geometry_cylindrical_test  parallel   physics  utilities dataset)

my_test(topology_structured_test    )  
target_link_libraries(topology_structured_test  parallel   physics  utilities dataset)

my_test(interpolator_test    )  
target_link_libraries(interpolator_test  parallel   physics  utilities dataset)
