

function(simpla_test name)

    add_executable(${name} ${ARGN})

    if (BUILD_SHARED_LIBS)
        set_target_properties(${name}
                PROPERTIES
                COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1 GTEST_HAS_TR1_TUPLE=0  ")
    endif ()

    target_link_libraries(${name} ${GTEST_BOTH_LIBRARIES} pthread)

    GTEST_ADD_TESTS(${name} "" ${ARGN})

#    ADD_DEPENDENCIES(${name} googletest)
    ADD_DEPENDENCIES(alltest ${name} )
endfunction()
 