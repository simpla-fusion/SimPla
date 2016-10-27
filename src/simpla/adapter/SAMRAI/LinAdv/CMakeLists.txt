INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})


GET_FILENAME_COMPONENT(PDAT_FORTDIR ${SAMRAI_SOURCE_DIR}/source/SAMRAI/pdat/fortran ABSOLUTE)

function(ADD_FORTRAN_SOURCES _ROOT name)
    GET_FILENAME_COMPONENT(ABS_PATH ${_ROOT}/${name}/fortran/ ABSOLUTE)
    MESSAGE(STATUS ${ABS_PATH})
    FILE(GLOB_RECURSE outfiles ${ABS_PATH}/*.f)
    FILE(GLOB_RECURSE ${name}_m4_SRC ${ABS_PATH}/*.m4)

    foreach (f ${${name}_m4_SRC})
        # first we might need to make the input file absolute
        GET_FILENAME_COMPONENT(f "${f}" ABSOLUTE)

        FILE(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
        STRING(REGEX REPLACE "\\.m4$" ".f" of "${CMAKE_CURRENT_SOURCE_DIR}/${rf}")
        LIST(APPEND outfiles "${of}")

        GET_FILENAME_COMPONENT(d "${of}" PATH)

        if (NOT IS_DIRECTORY "${d}")
            FILE(MAKE_DIRECTORY "${d}")
        endif (NOT IS_DIRECTORY "${d}")

        ADD_CUSTOM_COMMAND(OUTPUT "${of}"
                COMMAND m4 ARGS -DFORTDIR=${ABS_PATH} -DPDAT_FORTDIR=${PDAT_FORTDIR} ${f} > ${of}
                DEPENDS "${f}"
                )
    endforeach (f)

    SET(${name}_f_SRC ${outfiles} PARENT_SCOPE)

endfunction(ADD_FORTRAN_SOURCES)

function(ADD_SAMRAI_EXE name _ROOT)

    if (NOT DEFINED ${name}_SRC)
        FILE(GLOB_RECURSE ${name}_SRC ${_ROOT}/*.C)
    endif ()
    ADD_FORTRAN_SOURCES(${_ROOT}/../ ${name})
    MESSAGE(STATUS ${${name}_f_SRC})
    ADD_EXECUTABLE(${name} "${${name}_SRC}" ${${name}_f_SRC})

    TARGET_LINK_LIBRARIES(${name} ${SAMRAI_LIBRARIES} ${MPI_LIBRARIES} ${HDF5_LIBRARIES})

endfunction()

ADD_SAMRAI_EXE(LinAdv ./)
