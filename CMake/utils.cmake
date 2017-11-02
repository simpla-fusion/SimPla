

function(simpla_test name)
    add_executable(${name} ${ARGN})
    if (BUILD_SHARED_LIBS)
        set_target_properties(${name}
                PROPERTIES
                COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1 GTEST_HAS_TR1_TUPLE=0  ")
    endif ()

    target_link_libraries(${name} ${GTEST_BOTH_LIBRARIES} pthread)
    GTEST_ADD_TESTS(${name} "" ${ARGN})
    #    ADD_DEPENDENCIES(${GetPrefix} googletest)
    #    ADD_DEPENDENCIES(alltest ${GetPrefix} )
endfunction()

MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    FOREACH (child ${children})
        IF (EXISTS ${curdir}/${child}/CMakeLists.txt)
            LIST(APPEND dirlist ${child})
        ENDIF ()
    ENDFOREACH ()
    SET(${result} ${dirlist})
ENDMACRO()