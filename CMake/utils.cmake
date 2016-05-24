
function(my_test name )

 add_executable(${name} ${name}.cpp ${ARGN})

  if (BUILD_SHARED_LIBS)
    set_target_properties(${name}
      PROPERTIES
      COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1 GTEST_HAS_TR1_TUPLE=0  ")
  endif()
  
target_link_libraries(${name} gtest gtest_main pthread )   
 
#GTEST_ADD_TESTS(${name}  " "   ${name}.cpp    ${ARGN} )  
 add_test(${name}  ${EXECUTABLE_OUTPUT_PATH}/${name}  )  

  ADD_DEPENDENCIES(${name} googletest) 
endfunction()
 

function(simpla_test name )

 add_executable(${name}   ${ARGN})

 if (BUILD_SHARED_LIBS)
    set_target_properties(${name}
      PROPERTIES
      COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1 GTEST_HAS_TR1_TUPLE=0  ")
 endif()
  
target_link_libraries(${name} gtest gtest_main pthread)   
 
#GTEST_ADD_TESTS(${name}  " "   ${name}.cpp    ${ARGN} )  
 add_test(${name}  ${EXECUTABLE_OUTPUT_PATH}/${name}  )
 ADD_DEPENDENCIES(${name} googletest)
endfunction()
 