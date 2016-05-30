#my_test(distributed_array_test    
#         update_ghosts.cpp 
#      )
#target_link_libraries(distributed_array_test parallel  utilities  )

simpla_test(multi_thread_test multi_thread_test.cpp)
target_link_libraries(multi_thread_test ${TBB_LIBRARIES})
