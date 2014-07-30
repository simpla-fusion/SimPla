my_test(topology_UniformArray_test    )  
target_link_libraries(topology_UniformArray_test  parallel   physics  utilities)


my_test(geometry_cartesian_test     )  
target_link_libraries(geometry_cartesian_test   parallel   physics  utilities )

my_test(geometry_cylindrical_test    )  
target_link_libraries(geometry_cylindrical_test  parallel   physics  utilities )


my_test(interpolator_test    )  
target_link_libraries(interpolator_test  parallel   physics  utilities )
