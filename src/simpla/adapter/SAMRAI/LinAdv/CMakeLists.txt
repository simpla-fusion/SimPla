INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})


function(ADD_SAMRAI_EXE name _ROOT)

    if (NOT DEFINED ${name}_SRC)
        FILE(GLOB_RECURSE ${name}_SRC ${_ROOT}/*.C)
    endif ()

    ADD_EXECUTABLE(${name} "${${name}_SRC}")

    TARGET_LINK_LIBRARIES(${name} ${SAMRAI_LIBRARIES} ${MPI_LIBRARIES} ${HDF5_LIBRARIES})

endfunction()

ADD_SAMRAI_EXE(LinAdv ./)
