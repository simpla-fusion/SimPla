cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0028 NEW)
cmake_policy(SET CMP0021 OLD)

PROJECT(SAMRAI CXX C Fortran)
SET(CMAKE_VERBOSE_MAKEFILE ON)


SET(PREFIX ${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}-${CMAKE_BUILD_TYPE})
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
#SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})
MESSAGE(STATUS "Using Fortran Compiler: ${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION}")
MESSAGE(STATUS "Using C Compiler: ${CMAKE_C_COMPILER_ID}  ${CMAKE_C_COMPILER_VERSION}")
MESSAGE(STATUS "Using C++ Compiler: ${CMAKE_CXX_COMPILER_ID}  ${CMAKE_CXX_COMPILER_VERSION}")

SET(SAMRAI_VERSION 3.11.2)
###################################################################################################
FIND_PACKAGE(Boost REQUIRED)
IF (Boost_FOUND)
    ADD_DEFINITIONS(-DOMPI_SKIP_MPICXX)
    SET(HAVE_BOOST 1)
    SET(HAVE_BOOST_HEADERS 1)
ENDIF (Boost_FOUND)

FIND_PACKAGE(OpenMP REQUIRED)
IF (OpenMP_FOUND)
    SET(HAVE_OPENMP 1)
ENDIF (OpenMP_FOUND)

FIND_PACKAGE(MPI REQUIRED)
IF (MPI_FOUND)
    ADD_DEFINITIONS(-DOMPI_SKIP_MPICXX)
    SET(HAVE_MPI 1)
ENDIF (MPI_FOUND)

FIND_PACKAGE(CUDA)
IF (CUDA_FOUND)
    SET(HAVE_CUDA 1)
ENDIF (CUDA_FOUND)

#SET(TBB_ROOT /pkg/tbb/20170604/)
#FIND_PACKAGE(TBB)
#IF (TBB_FOUND)
#    SET(HAVE_TBB 1)
#ENDIF (TBB_FOUND)

find_package(PkgConfig)
IF (PKG_CONFIG_FOUND)
    pkg_search_module(tbb REQUIRED tbb)
    IF (tbb_FOUND)
        SET(HAVE_TBB 1)
        MESSAGE("${tbb_LIBRARIES}")
    ENDIF (tbb_FOUND)
ENDIF (PKG_CONFIG_FOUND)

SET(HDF5_PREFER_PARALLEL OFF)
SET(HDF5_USE_STATIC_LIBRARIES OFF)
FIND_PACKAGE(HDF5 1.8 COMPONENTS C HL REQUIRED)
IF (HDF5_FOUND)
    SET(HAVE_HDF5 1)
ENDIF (HDF5_FOUND)

FIND_PACKAGE(BLAS)
IF (BLAS_FOUND)
    SET(HAVE_BLAS 1)
ENDIF (BLAS_FOUND)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/config/SAMRAI_config.h.in TMP_CONFIG)
string(REGEX REPLACE "#undef SAMRAI_F77_FUNC[^_]" "#define SAMRAI_F77_FUNC(name,NAME) name ## _" TMP_CONFIG "${TMP_CONFIG}")
string(REGEX REPLACE "#undef SAMRAI_F77_FUNC_" "#define SAMRAI_F77_FUNC_(name,NAME) name ## _" TMP_CONFIG "${TMP_CONFIG}")
string(REGEX REPLACE "#undef ([a-zA-Z0-9_]+)" "#cmakedefine \\1 @\\1@" TMP_CONFIG "${TMP_CONFIG}")
string(REPLACE "MAX_DIM_VAL = SAMRAI_MAXIMUM_DIMENSION" "MAX_DIM_VAL = SAMRAI_MAXIMUM_DIMENSION\;" TMP_CONFIG "${TMP_CONFIG}")



SET(ENABLE_SAMRAI_TIMERS 1)
SET(SAMRAI_MAXIMUM_DIMENSION 3)
#SET(SAMRAI_F77_FUNC "(name,NAME) name ## _")
#SET(SAMRAI_F77_FUNC_ "(name,NAME) name ## _")
#SET(LACKS_STRSTREAM ON)
SET(IOMANIP_HEADER_FILE "<iomanip>")
SET(IOSTREAM_HEADER_FILE "<iostream>")
SET(STL_SSTREAM_HEADER_FILE "<sstream>")

include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)
include(CheckCXXSymbolExists)
CHECK_INCLUDE_FILE_CXX(cmath HAVE_CMATH)
CHECK_INCLUDE_FILE_CXX(ctime HAVE_CTIME)
CHECK_INCLUDE_FILE_CXX(sstream HAVE_SSTREAM)


CHECK_INCLUDE_FILE(inttypes.h HAVE_INTTYPES_H)
CHECK_INCLUDE_FILE(malloc.h HAVE_MALLOC_H)
CHECK_INCLUDE_FILE(memory.h HAVE_MEMORY_H)
CHECK_INCLUDE_FILE(stdint.h HAVE_STDINT_H)
CHECK_INCLUDE_FILE(stdlib.h HAVE_STDLIB_H)
CHECK_INCLUDE_FILE(strings.h HAVE_STRINGS_H)
CHECK_INCLUDE_FILE(string.h HAVE_STRING_H)
CHECK_INCLUDE_FILE(sys/stat.h HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILE(sys/times.h HAVE_SYS_TIMES_H)
CHECK_INCLUDE_FILE(unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILE(sys/types.h HAVE_SYS_TYPES_H)

check_symbol_exists(mallinfo malloc.h HAVE_MALLINFO)


check_symbol_exists(DBL_MAX float.h HAVE_DBL_MAX)
IF (NOT HAVE_DBL_MAX)
    SET(DBL_MAX_IS_BROKEN 1)
ENDIF (NOT HAVE_DBL_MAX)

check_symbol_exists(DBL_SNAN float.h HAVE_DBL_SNAN)
IF (NOT HAVE_DBL_SNAN)
    SET(DBL_SNAN_IS_BROKEN 1)
ENDIF (NOT HAVE_DBL_SNAN)

check_symbol_exists(FLT_MAX float.h HAVE_FLT_MAX)
IF (NOT HAVE_FLT_MAX)
    SET(FLT_MAX_IS_BROKEN 1)
ENDIF (NOT HAVE_FLT_MAX)

check_symbol_exists(FLT_SNAN float.h HAVE_FLT_SNAN)
IF (NOT HAVE_FLT_SNAN)
    SET(FLT_SNAN_IS_BROKEN 1)
ENDIF (NOT HAVE_FLT_SNAN)

check_cxx_symbol_exists(isnan cmath HAVE_CMATH_ISNAN)

CHECK_CXX_SOURCE_COMPILES(
        "#include <sstream>
        int main(){
         using namespace std;
         ostringstream oss; oss << 10;
        }"
        HAVE_OSTRINGSTREAM)
IF (NOT HAVE_OSTRINGSTREAM)
    SET(OSTRINGSTREAM_TYPE_IS_BROKEN 1)
ENDIF (NOT HAVE_OSTRINGSTREAM)

#CHECK_CXX_SOURCE_COMPILES(
#        "#include <strstream>
#        int main(){
#         using namespace std;
#         ostrstream oss; oss << 10;
#        }"
#        HAVE_OSTRSTREAM)
#check_cxx_symbol_exists(std::ostrstream  strstream HAVE_OSTRSTREAM)
#IF(NOT HAVE_OSTRSTREAM)
SET(OSTRSTREAM_TYPE_IS_BROKEN 1)
#ENDIF(NOT HAVE_OSTRSTREAM)


STRING(CONFIGURE "${TMP_CONFIG}" OUTPUT_CONFIG @ONLY)

#CONFIGURE_FILE(${CMAKE_CURRENT_BINARY_DIR}/SAMRAI_config.h.in
#        ${CMAKE_CURRENT_BINARY_DIR}/include/SAMRAI/SAMRAI_config.h @ONLY)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/include/SAMRAI/SAMRAI_config.h ${OUTPUT_CONFIG})
###################################################################################################

SET(CMAKE_CXX_FLAGS "-fPIC -Wno-deprecated -Wpotentially-evaluated-expression")
SET(CMAKE_Fortran_FLAGS  "-fPIC")

SET(SAMRAI_INCLUDE_DIRS ${SAMRAI_BINARY_DIR}/include/ ${SAMRAI_SOURCE_DIR}/source/)


SET(algs_OBJS
        TimeRefinementIntegrator
        TimeRefinementIntegratorConnectorWidthRequestor
        TimeRefinementLevelStrategy
        HyperbolicPatchStrategy
        HyperbolicLevelIntegrator
        ImplicitEquationStrategy
        ImplicitIntegrator
        MethodOfLinesIntegrator
        MethodOfLinesPatchStrategy
        OuternodeSumTransaction
        OuternodeSumTransactionFactory
        PatchBoundaryNodeSum
        OuteredgeSumTransaction
        OuteredgeSumTransactionFactory
        PatchBoundaryEdgeSum
        )
SET(appu_OBJS
        BoundaryUtilityStrategy
        VisItDataWriter
        VisDerivedDataStrategy
        VisMaterialsDataStrategy
        CartesianBoundaryUtilities2
        CartesianBoundaryUtilities3
        )
SET(geom_OBJS
        GridGeometry
        CartesianPatchGeometry
        CartesianGridGeometry
        CartesianCellDoubleLinearRefine
        CartesianCellDoubleConservativeLinearRefine
        CartesianCellDoubleWeightedAverage
        CartesianOutersideDoubleWeightedAverage
        CartesianSideDoubleWeightedAverage
        CartesianSideDoubleConservativeLinearRefine
        CartesianEdgeDoubleWeightedAverage
        CartesianEdgeDoubleConservativeLinearRefine
        CartesianNodeDoubleLinearRefine
        CartesianOuterfaceDoubleWeightedAverage
        CartesianFaceDoubleWeightedAverage
        CartesianFaceDoubleConservativeLinearRefine
        CartesianCellFloatLinearRefine
        CartesianCellFloatWeightedAverage
        CartesianCellFloatConservativeLinearRefine
        CartesianSideFloatConservativeLinearRefine
        CartesianSideFloatWeightedAverage
        CartesianEdgeFloatConservativeLinearRefine
        CartesianEdgeFloatWeightedAverage
        CartesianNodeFloatLinearRefine
        CartesianOuterfaceFloatWeightedAverage
        CartesianFaceFloatWeightedAverage
        CartesianFaceFloatConservativeLinearRefine
        CartesianCellComplexLinearRefine
        CartesianCellComplexWeightedAverage
        CartesianCellComplexConservativeLinearRefine
        CartesianSideComplexWeightedAverage
        CartesianEdgeComplexWeightedAverage
        CartesianNodeComplexLinearRefine
        CartesianOuterfaceComplexWeightedAverage
        CartesianFaceComplexWeightedAverage
        )

SET(hier_OBJS
        BaseGridGeometry
        BlockId
        BoxContainer
        CoarseFineBoundary
        LocalId
        PatchBoundaries
        PatchHierarchy
        HierarchyNeighbors
        FlattenedHierarchy
        PatchLevel
        PatchLevelFactory
        PeriodicId
        SingularityFinder
        PatchDataRestartManager
        VariableDatabase
        Variable
        PatchDescriptor
        PatchFactory
        PatchDataFactory
        Patch
        PatchGeometry
        PatchData
        BoxLevelConnectorUtils
        BoxContainerUtils
        BaseConnectorAlgorithm
        OverlapConnectorAlgorithm
        MappingConnectorAlgorithm
        Connector
        MappingConnector
        ConnectorStatistics
        BoxLevelHandle
        PeriodicShiftCatalog
        RealBoxConstIterator
        BoxContainerSingleBlockIterator
        BoxContainerSingleOwnerIterator
        BoxTree
        MultiblockBoxTree
        AssumedPartitionBox
        AssumedPartition
        BoxLevel
        BoxLevelStatistics
        PersistentOverlapConnectors
        BoxNeighborhoodCollection
        BoxOverlap
        BoxGeometry
        BoundaryBoxUtils
        BoxUtilities
        BoundaryBox
        BoundaryLookupTable
        Box
        Index
        IntVector
        GlobalId
        BoxId
        ProcessorMapping
        ComponentSelector
        VariableContext
        TransferOperatorRegistry
        CoarsenOperator
        RefineOperator
        TimeInterpolateOperator
        Transformation
        UncoveredBoxIterator
        )
SET(math_OBJS
        PatchSideDataOpsInteger
        PatchFaceDataOpsInteger
        PatchEdgeDataOpsInteger
        PatchCellDataOpsInteger
        PatchNodeDataOpsInteger
        PatchSideDataOpsComplex
        PatchNodeDataNormOpsComplex
        PatchSideDataNormOpsComplex
        PatchFaceDataOpsComplex
        PatchFaceDataNormOpsComplex
        PatchNodeDataOpsComplex
        PatchEdgeDataOpsComplex
        PatchEdgeDataNormOpsComplex
        PatchCellDataOpsComplex
        PatchCellDataNormOpsComplex
        HierarchyCellDataOpsInteger
        HierarchyEdgeDataOpsInteger
        HierarchyFaceDataOpsInteger
        HierarchySideDataOpsInteger
        HierarchyNodeDataOpsInteger
        HierarchyDataOpsManager
        HierarchyDataOpsInteger
        HierarchyCellDataOpsComplex
        HierarchyEdgeDataOpsComplex
        HierarchyFaceDataOpsComplex
        HierarchySideDataOpsComplex
        HierarchyNodeDataOpsComplex
        HierarchyDataOpsComplex
        ArrayDataNormOpsComplex
        ArrayDataNormOpsInteger
        )
SET(mesh_OBJS
        SpatialKey
        BoxGeneratorStrategy
        GriddingAlgorithm
        GriddingAlgorithmConnectorWidthRequestor
        StandardTagAndInitStrategy
        StandardTagAndInitialize
        StandardTagAndInitializeConnectorWidthRequestor
        TagAndInitializeStrategy
        BalanceUtilities
        TreeLoadBalancer
        GraphLoadBalancer
        ChopAndPackLoadBalancer
        CascadePartitioner
        CascadePartitionerTree
        LoadBalanceStrategy
        BalanceBoxBreaker
        BoxTransitSet
        VoucherTransitLoad
        TransitLoad
        GriddingAlgorithmStrategy
        BergerRigoutsos
        BergerRigoutsosNode
        MultiblockGriddingTagger
        TileClustering
        PartitioningParams
        )
SET(pdat_OBJS
        CellComplexLinearTimeInterpolateOp
        CellDoubleLinearTimeInterpolateOp
        CellFloatLinearTimeInterpolateOp
        EdgeComplexLinearTimeInterpolateOp
        EdgeDoubleLinearTimeInterpolateOp
        EdgeFloatLinearTimeInterpolateOp
        FaceComplexLinearTimeInterpolateOp
        FaceDoubleLinearTimeInterpolateOp
        FaceFloatLinearTimeInterpolateOp
        NodeComplexLinearTimeInterpolateOp
        NodeDoubleLinearTimeInterpolateOp
        NodeFloatLinearTimeInterpolateOp
        OuterfaceComplexLinearTimeInterpolateOp
        OuterfaceDoubleLinearTimeInterpolateOp
        OuterfaceFloatLinearTimeInterpolateOp
        OutersideComplexLinearTimeInterpolateOp
        OutersideDoubleLinearTimeInterpolateOp
        OutersideFloatLinearTimeInterpolateOp
        SideComplexLinearTimeInterpolateOp
        SideDoubleLinearTimeInterpolateOp
        SideFloatLinearTimeInterpolateOp
        CellComplexConstantRefine
        CellDoubleConstantRefine
        CellFloatConstantRefine
        CellIntegerConstantRefine
        EdgeComplexConstantRefine
        EdgeDoubleConstantRefine
        EdgeFloatConstantRefine
        EdgeIntegerConstantRefine
        FaceComplexConstantRefine
        FaceDoubleConstantRefine
        FaceFloatConstantRefine
        FaceIntegerConstantRefine
        OuterfaceComplexConstantRefine
        OuterfaceDoubleConstantRefine
        OuterfaceFloatConstantRefine
        OuterfaceIntegerConstantRefine
        SideComplexConstantRefine
        SideDoubleConstantRefine
        SideFloatConstantRefine
        SideIntegerConstantRefine
        NodeComplexInjection
        NodeDoubleInjection
        NodeFloatInjection
        NodeIntegerInjection
        OuternodeDoubleInjection
        CellIterator
        EdgeIterator
        FaceIterator
        NodeIterator
        SideIterator
        CellIndex
        EdgeIndex
        FaceIndex
        NodeIndex
        SideIndex
        CellGeometry
        EdgeGeometry
        FaceGeometry
        NodeGeometry
        OuteredgeGeometry
        OuterfaceGeometry
        OuternodeGeometry
        OutersideGeometry
        SideGeometry
        CellOverlap
        EdgeOverlap
        FaceOverlap
        NodeOverlap
        SideOverlap
        ArrayDataIterator
        FirstLayerCellVariableFillPattern
        FirstLayerCellNoCornersVariableFillPattern
        FirstLayerEdgeVariableFillPattern
        FirstLayerNodeVariableFillPattern
        FirstLayerSideVariableFillPattern
        SecondLayerNodeVariableFillPattern
        SecondLayerNodeNoCornersVariableFillPattern
        SparseData
        SparseDataFactory
        SparseDataVariable
        IntegerAttributeId
        DoubleAttributeId
        )

SET(solv_OBJS
        PoissonSpecifications
        SNESAbstractFunction
        CVODEAbstractFunctions
        CVODESolver
        KINSOLAbstractFunctions
        KINSOLSolver
        solv_NVector
        SundialsAbstractVector
        FACOperatorStrategy
        FACPreconditioner
        NonlinearSolverStrategy
        SNES_SAMRAIContext
        KINSOL_SAMRAIContext
        CartesianRobinBcHelper
        CellPoissonFACOps
        CellPoissonFACSolver
        CellPoissonHypreSolver
        GhostCellRobinBcCoefs
        LocationIndexRobinBcCoefs
        RobinBcCoefStrategy
        SimpleCellRobinBcCoefs
        Sundials_SAMRAIVector
        )

SET(tbox_OBJS
        ArraySpecial
        AsyncCommGroup
        AsyncCommStage
        BalancedDepthFirstTree
        BreadthFirstRankTree
        CenteredRankTree
        Clock
        CommGraphWriter
        Database
        DatabaseBox
        DatabaseFactory
        Dimension
        Grammar
        HDFDatabase
        HDFDatabaseFactory
        IEEE
        InputManager
        Logger
        MathUtilitiesSpecial
        MemoryDatabase
        MemoryDatabaseFactory
        MemoryUtilities
        MessageStream
        NullDatabase
        PIO
        ParallelBuffer
        Parser
        RankGroup
        RankTreeStrategy
        ReferenceCounter
        RestartManager
        SAMRAIManager
        SAMRAI_MPI
        Scanner
        Schedule
        Serializable
        SiloDatabase
        SiloDatabaseFactory
        StartupShutdownManager
        StatTransaction
        Statistic
        Statistician
        Timer
        TimerManager
        Tracer
        Transaction
        Utilities)

SET(xfer_OBJS
        BoxGeometryVariableFillPattern
        PatchInteriorVariableFillPattern
        PatchLevelFullFillPattern
        PatchLevelBorderFillPattern
        PatchLevelBorderAndInteriorFillPattern
        PatchLevelEnhancedFillPattern
        PatchLevelInteriorFillPattern
        PatchLevelFillPattern
        VariableFillPattern
        RefineAlgorithm
        CoarsenAlgorithm
        RefineSchedule
        RefineScheduleConnectorWidthRequestor
        CoarsenSchedule
        RefineTimeTransaction
        RefineCopyTransaction
        StandardRefineTransactionFactory
        RefineTransactionFactory
        RefinePatchStrategy
        RefineClasses
        CoarsenCopyTransaction
        StandardCoarsenTransactionFactory
        CoarsenTransactionFactory
        CoarsenPatchStrategy
        CoarsenClasses
        SingularityPatchStrategy
        CompositeBoundaryAlgorithm
        CompositeBoundarySchedule
        )

INCLUDE_DIRECTORIES(${SAMRAI_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${MPI_C_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIRS})

find_program(M4_EXECUTABLE m4 DOC "The M4 macro processor")

if (NOT M4_EXECUTABLE)
    message(SEND_ERROR "Failed to find the M4 macro processor.")
endif (NOT M4_EXECUTABLE)

SET(PDAT_FORTDIR ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/pdat/fortran)

function(GET_SOURCE_LIST SRC  PREFIX MOD_NAME )
    MESSAGE("Add MODULE : ${PREFIX}")
    SET(C_SRC)
    SET(F_SRC)
    IF (${MOD_NAME}_OBJS)
        FOREACH (OBJ ${${MOD_NAME}_OBJS})
            SET(src_file ${PREFIX}/${OBJ}.C)
            IF (EXISTS ${src_file})
                LIST(APPEND C_SRC ${src_file})
            ENDIF ()
        ENDFOREACH ()
    ELSE (${MOD_NAME}_OBJS)
        FILE(GLOB C_SRC ${PREFIX}/*.C)
    ENDIF (${MOD_NAME}_OBJS)

    IF (IS_DIRECTORY ${PREFIX}/fortran)
        SET(M4_FLAGS -DFORTDIR=${PREFIX}/fortran -DPDAT_FORTDIR=${PDAT_FORTDIR})
        FILE(GLOB m4_SRC ${PREFIX}/fortran/*.m4)
#        ADD_M4_F_SOURCES(F_SRC "${m4_FLAGS}" ${m4_SRC})
        foreach (f ${m4_SRC})
            # first we might need to make the input file absolute
            get_filename_component(f "${f}" ABSOLUTE)
            # get the relative path of the file to the current source dir
            file(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
            # strip the .m4 off the end if present and prepend the current  binary dir
            string(REGEX REPLACE "\\.m4$" ".f" of "${CMAKE_CURRENT_BINARY_DIR}/${rf}")
            # append the output file to the list of outputs
            # create the output directory if it doesn't exist
            get_filename_component(d "${of}" PATH)
            if (NOT IS_DIRECTORY "${d}")
                file(MAKE_DIRECTORY "${d}")
            endif (NOT IS_DIRECTORY "${d}")
            # now add the custom command to generate the output file
            add_custom_command(OUTPUT "${of}"
                    COMMAND ${M4_EXECUTABLE} ${M4_FLAGS} "${f}" > "${of}"
                    DEPENDS "${f}"
                    )
            list(APPEND F_SRC "${of}")
        endforeach (f)
        FILE(GLOB f_src ${PREFIX}/fortran/*.f)
        FOREACH (f ${f_src})
            LIST(APPEND F_SRC ${f})
        ENDFOREACH ()
    ENDIF ()
#    set(${C_SRC} PARENT_SCOPE)
#    set(${F_SRC} PARENT_SCOPE)
    LIST(APPEND SRC ${C_SRC})
    LIST(APPEND SRC ${F_SRC})
    set(${SRC} PARENT_SCOPE)
endfunction(GET_SOURCE_LIST)

SET(SAMRAI_LIBRARIES algs appu geom  math mesh pdat solv tbox xfer hier)
FOREACH (lib_name ${SAMRAI_LIBRARIES})
    GET_SOURCE_LIST(${lib_name}_SRC ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/${lib_name}  ${lib_name} )
    add_library(${lib_name} ${${lib_name}_SRC}  )
ENDFOREACH ()

GET_SOURCE_LIST(ConvDiff_SRC ${CMAKE_CURRENT_SOURCE_DIR}/source/test/applications/ConvDiff ConvDiff)
ADD_EXECUTABLE(ConvDiff ${ConvDiff_SRC})
target_link_libraries(ConvDiff ${SAMRAI_LIBRARIES} ${MPI_C_LIBRARIES} ${HDF5_LIBRARIES})