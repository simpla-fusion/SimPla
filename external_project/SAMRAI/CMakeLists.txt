cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0028 NEW)


PROJECT(SAMRAI CXX C Fortran)


FIND_PACKAGE(MPI)

IF (MPI_FOUND)
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
    ADD_DEFINITIONS(-DUSE_MPI -DMPICH_SKIP_MPICXX)
    ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX)
ENDIF (MPI_FOUND)

SET(HDF5_PREFER_PARALLEL ON)
SET(HDF5_USE_STATIC_LIBRARIES OFF)

FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL)

IF (HDF5_FOUND)
    INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
    IF (HDF5_IS_PARALLEL)
        ADD_DEFINITIONS(-DUSE_PARALLEL_IO -DH5_HAVE_PARALLEL)
    ENDIF (HDF5_IS_PARALLEL)
ENDIF (HDF5_FOUND)


SET(SAMRAI_DIR ./)


INCLUDE_DIRECTORIES(${SAMRAI_DIR}/include)
INCLUDE_DIRECTORIES(${SAMRAI_DIR}/source)

SET(xfer_SRC
        ${SAMRAI_DIR}/source/SAMRAI/xfer/BoxGeometryVariableFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/PatchInteriorVariableFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/PatchLevelFullFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/PatchLevelBorderFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/PatchLevelBorderAndInteriorFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/PatchLevelEnhancedFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/PatchLevelInteriorFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/PatchLevelFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/VariableFillPattern.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefineAlgorithm.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/CoarsenAlgorithm.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefineSchedule.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefineScheduleConnectorWidthRequestor.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/CoarsenSchedule.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefineTimeTransaction.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefineCopyTransaction.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/StandardRefineTransactionFactory.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefineTransactionFactory.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefinePatchStrategy.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/RefineClasses.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/CoarsenCopyTransaction.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/StandardCoarsenTransactionFactory.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/CoarsenTransactionFactory.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/CoarsenPatchStrategy.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/CoarsenClasses.C
        ${SAMRAI_DIR}/source/SAMRAI/xfer/SingularityPatchStrategy.C
        )


# - Pass a list of files through the M4 macro processor
#
# ADD_M4_SOURCES( OUTVAR source1 ... sourceN )
#
#  OUTVAR  A list containing all the output file names, suitable
#          to be passed to add_executable or add_library.
#
# If the source files have a .m4 suffix it is stripped from the output
# file name. The output files are placed in the same relative location
# to CMAKE_CURRENT_BINARY_DIR as they are to CMAKE_CURRENT_SOURCE_DIR.
#
# Example:
#  add_m4_sources( SRCS src/test1.cxx.m4 src/test2.cxx.m4 )
#  add_executable( test ${SRCS} )
function(ADD_M4_SOURCES OUTVAR)
    set(outfiles)
    foreach (f ${ARGN})
        # first we might need to make the input file absolute
        get_filename_component(f "${f}" ABSOLUTE)
        #if( NOT IS_ABSOLUTE "${f}" )
        #  set( f "${CMAKE_CURRENT_SOURCE_DIR}/${f}" )
        #endif( NOT IS_ABSOLUTE "${f}" )
        # get the relative path of the file to the current source dir
        file(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
        # strip the .m4 off the end if present and prepend the current binary dir string( REGEX REPLACE "\\.m4$" "" of "$ {CMAKE_CURRENT_BINARY_DIR}/${rf}" )
        # append the output file to the list of outputs
        list(APPEND outfiles "${of}")
        # create the output directory if it doesn't exist
        get_filename_component(d "${of}" PATH)

        # now add the custom command to generate the output file
        add_custom_command(OUTPUT "${of}"
                COMMAND m4 ARGS -P -s "${f}" > "${of}"
                DEPENDS "${f}"
                )
    endforeach (f)
    # set the output list in the calling scope
    MESSAGE(${OUTVAR})
    set(${OUTVAR} ${outfiles} PARENT_SCOPE)
endfunction(ADD_M4_SOURCES)

function(SAMRAI_build_lib _ROOT name)
    if (NOT DEFINED ${name}_SRC)
        FILE(GLOB_RECURSE ${name}_SRC ${_ROOT}/${name}/*.C)
    endif ()

    if (NOT DEFINED ${name}_f_SRC)
        FILE(GLOB_RECURSE ${name}_m4_SRC ${_ROOT}/${name}/fortran/*.m4)
        ADD_M4_SOURCES(${name}_f_SRC ${${name}_m4_SRC})
    endif ()

    ADD_LIBRARY(SAMRAI_${name} ${${name}_SRC} ${${name}_f_SRC})

endfunction()

function(SAMRAI_build_exe _ROOT name)

    if (NOT DEFINED ${name}_SRC)
        FILE(GLOB_RECURSE ${name}_SRC ${_ROOT}/${name}/*.C)
    endif ()

    if (NOT DEFINED ${name}_f_SRC)
        FILE(GLOB_RECURSE ${name}_f_SRC ${_ROOT}/${name}/fortran/*.f)
    endif ()

    ADD_EXECUTABLE(${name}.exe ${${name}_SRC} ${${name}_f_SRC})

    TARGET_LINK_LIBRARIES(${name}.exe ${SAMRAI_LIBRARIES} ${MPI_LIBRARIES} ${HDF5_LIBRARIES})

endfunction()


SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI appu)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI algs)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI solv)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI geom)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI mesh)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI math)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI pdat)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI xfer)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI hier)
SAMRAI_build_lib(${SAMRAI_DIR}/source/SAMRAI tbox)
SAMRAI_build_lib(${SAMRAI_DIR}/source/test testlib)


SET(SAMRAI_LIBRARIES
        SAMRAI_testlib
        SAMRAI_appu
        SAMRAI_algs
        SAMRAI_solv
        SAMRAI_geom
        SAMRAI_mesh
        SAMRAI_math
        SAMRAI_pdat
        SAMRAI_xfer
        SAMRAI_hier
        SAMRAI_tbox
        )

SAMRAI_build_exe(${SAMRAI_DIR}/source/test MblkLinAdv)