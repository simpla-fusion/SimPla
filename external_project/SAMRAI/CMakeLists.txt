cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0028 NEW)

PROJECT(SAMRAI CXX C Fortran)

SET(CMAKE_VERBOSE_MAKEFILE ON)

FIND_PACKAGE(MPI)

IF (MPI_FOUND)
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
    ADD_DEFINITIONS(-DUSE_MPI -DMPICH_SKIP_MPICXX)
    ADD_DEFINITIONS(-DUSE_MPI -DOMPI_SKIP_MPICXX)
    SET(HAVE_MPI 1)
ENDIF (MPI_FOUND)

SET(HDF5_PREFER_PARALLEL ON)
SET(HDF5_USE_STATIC_LIBRARIES OFF)

FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL)

IF (HDF5_FOUND)
    INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})
    IF (HDF5_IS_PARALLEL)
        ADD_DEFINITIONS(-DUSE_PARALLEL_IO -DH5_HAVE_PARALLEL)
    ENDIF (HDF5_IS_PARALLEL)
    SET(HAVE_HDF5 1)
ENDIF (HDF5_FOUND)

FIND_PACKAGE(Boost)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    SET(HAVE_BOOST_HEADERS 1)
ENDIF (Boost_FOUND)

FIND_PACKAGE(BLAS)
IF (BLAS_FOUND)
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIRS})
    SET(HAVE_BLAS 1)
ENDIF (BLAS_FOUND)

FIND_PACKAGE(LAPACK)
IF (LAPACK_FOUND)
    INCLUDE_DIRECTORIES(${LAPACK_INCLUDE_DIRS})
    SET(HAVE_LAPACK 1)
ENDIF (LAPACK_FOUND)

SET(OPT_BUILD 1)

GET_FILENAME_COMPONENT(SAMRAI_DIR /home/salmon/workspace/SAMRAI-master ABSOLUTE)

SET(SAMRAI_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/SAMRAI)
if (NOT IS_DIRECTORY "${SAMRAI_INCLUDE_DIR}")
    FILE(MAKE_DIRECTORY "${SAMRAI_INCLUDE_DIR}")
endif (NOT IS_DIRECTORY "${SAMRAI_INCLUDE_DIR}")

SET(DBL_SNAN_IS_BROKEN 1)
SET(DEBUG_CHECK_ASSERTIONS 1)
SET(DEBUG_CHECK_DIM_ASSERTIONS 1)
SET(DEBUG_INITIALIZE_UNDEFINED 1)
SET(HAVE_MALLINFO 1)
SET(FLT_SNAN_IS_BROKEN 1)
SET(HAVE_CTIME 1)
SET(HAVE_EXCEPTION_HANDLING 1)
SET(HAVE_IOMANIP_LEFT 1)
SET(ENABLE_SAMRAI_TIMERS 1)
SET(HAVE_ISO_SSTREAM 1)
SET(HAVE_MEMBER_FUNCTION_SPECIALIZATION 1)
SET(HAVE_OPENMP 1)
SET(HAVE_NAMESPACE 1)
SET(HAVE_NEW_PLACEMENT_OPERATOR 1)
SET(OSTRSTREAM_TYPE_IS_BROKEN 1)
SET(HAVE_TEMPLATE_COMPLEX 1)
SET(HAVE_STANDARD_STATIC_DATA_SPECIALIZATION 1)
SET(HAVE_STATIC_DATA_INSTANTIATION 1)
SET(INCLUDE_DEPRECATED 2)
SET(SAMRAI_MAXIMUM_DIMENSION 3)
SET(IOMANIP_HEADER_FILE <iomanip>)
SET(IOSTREAM_HEADER_FILE <iostream>)

SET(LACKS_PROPER_XDR_HEADER 1)
SET(LACKS_SUNDIALS 1)
SET(LACKS_TAU 1)
SET(LACKS_VAMPIR 1)
SET(LACKS_X11 1)
SET(NAN_IS_BROKEN 1)
SET(OSTRSTREAM_TYPE_IS_BROKEN 1)
SET(SAMRAI_MAXIMUM_DIMENSION 3)
SET(STDC_HEADERS 1)
SET(STL_SSTREAM_HEADER_FILE <sstream>)
INCLUDE(CheckIncludeFiles)
CHECK_INCLUDE_FILE("sstream" HAVE_SSTREAM)
CHECK_INCLUDE_FILE(<cmath> HAVE_CMATH)
SET(HAVE_CMATH_ISNAN 1)

CHECK_INCLUDE_FILE("inttypes.h" HAVE_INTTYPES_H)

CHECK_INCLUDE_FILE("malloc.h" HAVE_MALLOC_H)
CHECK_INCLUDE_FILE("memory.h" HAVE_MEMORY_H)
CHECK_INCLUDE_FILE("sstream" HAVE_SSTREAM)
CHECK_INCLUDE_FILE("stdint.h" HAVE_STDINT_H)
CHECK_INCLUDE_FILE("stdlib.h" HAVE_STDLIB_H)
CHECK_INCLUDE_FILE("strings.h" HAVE_STRINGS_H)
CHECK_INCLUDE_FILE("string.h" HAVE_STRING_H)
CHECK_INCLUDE_FILE("sys/stat.h" HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILE("sys/times.h" HAVE_SYS_TIMES_H)
CHECK_INCLUDE_FILE("sys/types.h" HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILE("unistd.h" HAVE_UNISTD_H)


SET(CMAKE_CURRENT_SOURCE_DIR ${SAMRAI_DIR})

execute_process(COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/config/SAMRAI_config.h.in ${SAMRAI_INCLUDE_DIR}/SAMRAI_config.h.in)
execute_process(COMMAND sed -i "s/#undef SAMRAI_F77_FUNC_/#define SAMRAI_F77_FUNC_\(name,NAME\) name ## _/g" ${SAMRAI_INCLUDE_DIR}/SAMRAI_config.h.in)
execute_process(COMMAND sed -i "s/^#undef SAMRAI_F77_FUNC$/#define SAMRAI_F77_FUNC\(name,NAME\) name ## _/g" ${SAMRAI_INCLUDE_DIR}/SAMRAI_config.h.in)
execute_process(COMMAND sed -i "s/#undef \\([_1-9a-zA-Z]*\\)/#cmakedefine \\1 @\\1@/g" ${SAMRAI_INCLUDE_DIR}/SAMRAI_config.h.in)
configure_file(${SAMRAI_INCLUDE_DIR}/SAMRAI_config.h.in ${SAMRAI_INCLUDE_DIR}/SAMRAI_config.h @ONLY)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/source)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/include)
SET(tbox_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/ArraySpecial.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/AsyncCommGroup.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/AsyncCommStage.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/BalancedDepthFirstTree.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/BreadthFirstRankTree.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/CenteredRankTree.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Clock.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/CommGraphWriter.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Database.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/DatabaseBox.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/DatabaseFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Dimension.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Grammar.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/HDFDatabase.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/HDFDatabaseFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/IEEE.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/InputManager.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Logger.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/MathUtilitiesSpecial.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/MemoryDatabase.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/MemoryDatabaseFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/MemoryUtilities.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/MessageStream.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/NullDatabase.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/PIO.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/ParallelBuffer.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Parser.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/RankGroup.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/RankTreeStrategy.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/ReferenceCounter.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/RestartManager.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/SAMRAIManager.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/SAMRAI_MPI.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Scanner.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Schedule.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Serializable.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/SiloDatabase.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/SiloDatabaseFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/StartupShutdownManager.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/StatTransaction.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Statistic.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Statistician.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Timer.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/TimerManager.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Tracer.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Transaction.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/tbox/Utilities.C
        )

SET(xfer_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/BoxGeometryVariableFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/PatchInteriorVariableFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/PatchLevelFullFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/PatchLevelBorderFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/PatchLevelBorderAndInteriorFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/PatchLevelEnhancedFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/PatchLevelInteriorFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/PatchLevelFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/VariableFillPattern.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefineAlgorithm.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/CoarsenAlgorithm.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefineSchedule.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefineScheduleConnectorWidthRequestor.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/CoarsenSchedule.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefineTimeTransaction.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefineCopyTransaction.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/StandardRefineTransactionFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefineTransactionFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefinePatchStrategy.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/RefineClasses.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/CoarsenCopyTransaction.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/StandardCoarsenTransactionFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/CoarsenTransactionFactory.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/CoarsenPatchStrategy.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/CoarsenClasses.C
        ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/xfer/SingularityPatchStrategy.C
        )


GET_FILENAME_COMPONENT(PDAT_FORTDIR ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/pdat/fortran ABSOLUTE)

function(ADD_FORTRAN_SOURCES _ROOT name)
    GET_FILENAME_COMPONENT(ABS_PATH ${_ROOT}/${name}/fortran/ ABSOLUTE)

    FILE(GLOB_RECURSE outfiles ${ABS_PATH}/*.f)
    FILE(GLOB_RECURSE ${name}_m4_SRC ${ABS_PATH}/*.m4)

    foreach (f ${${name}_m4_SRC})
        # first we might need to make the input file absolute
        GET_FILENAME_COMPONENT(f "${f}" ABSOLUTE)

        FILE(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
        STRING(REGEX REPLACE "\\.m4$" ".f" of "${CMAKE_CURRENT_BINARY_DIR}/${rf}")
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

function(SAMRAI_build_lib _ROOT name)
    if (NOT DEFINED ${name}_SRC)
        FILE(GLOB_RECURSE ${name}_SRC ${_ROOT}/${name}/*.C)
    endif ()
    ADD_FORTRAN_SOURCES(${_ROOT} ${name})
    ADD_LIBRARY(SAMRAI_${name} ${${name}_SRC} ${${name}_f_SRC})
endfunction()

function(SAMRAI_build_exe name _ROOT libs)

    if (NOT DEFINED ${name}_SRC)
        FILE(GLOB_RECURSE ${name}_SRC ${_ROOT}/*.C)
    endif ()

    ADD_EXECUTABLE(${name} ${${name}_SRC})

    TARGET_LINK_LIBRARIES(${name} ${libs} ${SAMRAI_LIBRARIES} ${MPI_LIBRARIES} ${HDF5_LIBRARIES})

endfunction()

SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI pdat)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI appu)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI algs)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI solv)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI geom)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI mesh)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI math)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI xfer)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI hier)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI tbox)
SAMRAI_build_lib(${CMAKE_CURRENT_SOURCE_DIR}/source/test testlib)

SET(SAMRAI_LIBRARIES
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

SAMRAI_build_exe(MblkLinAdv ${CMAKE_CURRENT_SOURCE_DIR}/source/test/MblkLinAdv SAMRAI_testlib)
