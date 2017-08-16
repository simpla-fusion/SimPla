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

SET(CMAKE_CXX_FLAGS "--std=c++14   -fPIC -Wno-deprecated -Wpotentially-evaluated-expression")
SET(CMAKE_Fortran_FLAGS "-fPIC")

SET(SAMRAI_INCLUDE_DIRS ${SAMRAI_BINARY_DIR}/include/ ${SAMRAI_SOURCE_DIR}/source/)

SET(EXTRA_SRC )

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
        ArrayData # inc header
        ArrayDataAccess # inc header
        ArrayDataOperationUtilities # inc header
        CellData # inc header
        CellDataFactory  # inc header
        CellVariable # inc header
        EdgeData # inc header
        EdgeDataFactory  # inc header
        EdgeVariable # inc header
        FaceData # inc header
        FaceDataFactory  # inc header
        FaceVariable # inc header
        SideData # inc header
        SideDataFactory  # inc header
        SideVariable # inc header
        NodeData # inc header
        NodeDataFactory  # inc header
        NodeVariable # inc header
        IndexData # inc header
        IndexDataFactory  # inc header
        IndexVariable # inc header
        OuteredgeData # inc header
        OuteredgeDataFactory  # inc header
        OuteredgeVariable # inc header
        OuterfaceData # inc header
        OuterfaceDataFactory  # inc header
        OuterfaceVariable # inc header
        OuternodeData # inc header
        OuternodeDataFactory  # inc header
        OuternodeVariable # inc header
        OutersideData # inc header
        OutersideDataFactory  # inc header
        OutersideVariable # inc header
        SumOperation # inc header
        CopyOperation # inc header
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
        Array # inc header
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

function(GET_SOURCE_LIST _C_SRC _F_SRC _C_HEAD PREFIX MOD_NAME)
    MESSAGE("Add MODULE : ${PREFIX}")
    SET(C_SRC)
    SET(F_SRC)
    SET(C_HEAD)

    FILE(GLOB C_HEAD ${PREFIX}/*.h)
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
        foreach (f ${m4_SRC})
            get_filename_component(f "${f}" ABSOLUTE)
            file(RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}")
            string(REGEX REPLACE "\\.m4$" ".f" of "${CMAKE_CURRENT_BINARY_DIR}/${rf}")
            get_filename_component(d "${of}" PATH)
            if (NOT IS_DIRECTORY "${d}")
                file(MAKE_DIRECTORY "${d}")
            endif (NOT IS_DIRECTORY "${d}")
            add_custom_command(OUTPUT "${of}"
                    COMMAND ${M4_EXECUTABLE} ${M4_FLAGS} "${f}" > "${of}"
                    DEPENDS "${f}"
                    )
            list(APPEND F_SRC "${of}")
        endforeach (f)
        FILE(GLOB f_src ${PREFIX}/fortran/*.f)
        LIST(APPEND F_SRC ${f_src})
    ENDIF ()


    LIST(APPEND _C_SRC ${C_SRC})
    LIST(APPEND _F_SRC ${F_SRC})
    LIST(APPEND _C_HEAD ${C_HEAD})

    set(${_C_SRC} PARENT_SCOPE)
    set(${_F_SRC} PARENT_SCOPE)
    set(${_C_HEAD} PARENT_SCOPE)

endfunction(GET_SOURCE_LIST)

SET(SAMRAI_MODS algs appu geom math mesh pdat solv tbox xfer hier)
FOREACH (mod_name ${SAMRAI_MODS})
    GET_SOURCE_LIST(${mod_name}_C_SRC ${mod_name}_F_SRC ${mod_name}_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI/${mod_name} ${mod_name})
    LIST(APPEND SAMRAI_LIBRARIES SAMRAI_${mod_name})
    add_library(SAMRAI_${mod_name} ${${mod_name}_C_SRC} ${${mod_name}_F_SRC} ${${mod_name}_HEAD} ${EXTRA_SRC})
ENDFOREACH ()

GET_SOURCE_LIST(ConvDiff_C_SRC ConvDiff_F_SRC ConvDiff_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/source/test/applications/ConvDiff ConvDiff)
ADD_LIBRARY(ConvDiff_f ${ConvDiff_F_SRC})
ADD_EXECUTABLE(ConvDiff_cpu ${ConvDiff_C_SRC} ${ConvDiff_HEAD})
target_link_libraries(ConvDiff_cpu ${SAMRAI_LIBRARIES} ${MPI_C_LIBRARIES} ${HDF5_LIBRARIES} ${CUDA_LIBRARIES} ConvDiff_f)

ADD_EXECUTABLE(ConvDiff_cuda ${ConvDiff_C_SRC} ${ConvDiff_HEAD} ${CMAKE_CURRENT_SOURCE_DIR}/source/test/applications/ConvDiff/cuda/ConvDiff.cuda.C)
set_target_properties(ConvDiff_cuda PROPERTIES COMPILE_FLAGS "  -xcuda -fno-limit-debug-info -DDEBUG_INITIALIZE_UNDEFINED")
target_link_libraries(ConvDiff_cuda ${SAMRAI_LIBRARIES} ${MPI_C_LIBRARIES} ${HDF5_LIBRARIES} ${CUDA_LIBRARIES})
######################################################################################################################




INSTALL(FILES  ${CMAKE_CURRENT_BINARY_DIR}/include/SAMRAI/SAMRAI_config.h DESTINATION include/SAMRAI)


INSTALL(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/COPYRIGHT
        ${CMAKE_CURRENT_SOURCE_DIR}/COPYING.LESSER
        ${CMAKE_CURRENT_SOURCE_DIR}/RELEASE-NOTES
        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
        DESTINATION .)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/source/SAMRAI
            DESTINATION include
            FILES_MATCHING REGEX "[a-zA-Z0-9_-]+.[hC]"
#            PERMISSIONS OWNER_READ GROUP_READ
        )

INSTALL(TARGETS ${SAMRAI_LIBRARIES} EXPORT SAMRAITargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include
        )

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SAMRAIConfigVersion.cmake"
        VERSION ${SAMRAI_VERSION}
        COMPATIBILITY AnyNewerVersion
)

export(EXPORT SAMRAITargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/SAMRAITargets.cmake"
        NAMESPACE Upstream::
        )

STRING(CONFIGURE "${TMP_CONFIG}" OUTPUT_CONFIG @ONLY)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/include/SAMRAI/SAMRAI_config.h ${OUTPUT_CONFIG})

SET(SAMRAI_FOUND TRUE)
SET(SAMRAI_DIR ${CMAKE_INSTALL_PREFIX})

LIST(APPEND SAMRAI_INCLUDE_DIRS_DEP  ${HDF5_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH} ${CMAKE_INSTALL_PREFIX}/include/)
LIST(APPEND SAMRAI_INCLUDE_DIRS_DEP   )

MESSAGE("MPI INCLUDE : " ${MPI_C_INCLUDE_PATH})
MESSAGE("SAMRAI_INCLUDE_DIRS   : " ${SAMRAI_INCLUDE_DIRS})


LIST(APPEND SAMRAI_LIBRARIES_DEP ${MPI_C_LIBRARIES} ${HDF5_LIBRARIES} ${CUDA_LIBRARIES})

set(ConfigPackageLocation lib/cmake/SAMRAI)


FOREACH (LIB_ ${SAMRAI_LIBRARIES_LIST})
    find_library(${LIB_}_FILE NAMES ${LIB_} PATHS ${SAMRAI_LIBRARY_DIRS})
    IF (NOT ${LIB_}_NOTFOUND)
        LIST(APPEND SAMRAI_LIBRARIES_ABS "${${LIB_}_FILE}")
    ENDIF (NOT ${LIB_}_NOTFOUND)
ENDFOREACH ()
configure_file(SAMRAIConfig.cmake.in
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SAMRAIConfig.cmake" @ONLY)

configure_file(SAMRAIConfigVersion.cmake.in
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SAMRAIConfigVersion.cmake" @ONLY)



install(EXPORT SAMRAITargets
        FILE
        SAMRAITargets.cmake
        NAMESPACE
        Upstream::
        DESTINATION
        ${ConfigPackageLocation}
        )
install(FILES
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SAMRAIConfig.cmake
        ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/SAMRAIConfigVersion.cmake
        DESTINATION
        ${ConfigPackageLocation}
        COMPONENT
        Devel
)
