  
function(my_test name library )

  add_executable(${name} ${name}.cpp ${ARGN})

  if (BUILD_SHARED_LIBS)
    set_target_properties(${name}
      PROPERTIES
      COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1")
  endif()
  
  target_link_libraries(${name} ${library} gtest_main gtest pthread)   
 
  add_test(${name}  ${name}  )
  
 
  ADD_DEPENDENCIES(${name} googletest) 
endfunction()
