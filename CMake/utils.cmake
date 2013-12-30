  
function(my_test name )

  add_executable(${name} ${name}.cpp ${ARGN})

  if (BUILD_SHARED_LIBS)
    set_target_properties(${name}
      PROPERTIES
      COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1 GTEST_HAS_TR1_TUPLE=0  ")
  endif()
  
  target_link_libraries(${name} gtest_main gtest pthread)   
 
  add_test(${name}  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}  )  
 
  ADD_DEPENDENCIES(${name} googletest) 
endfunction()
