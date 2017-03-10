# - Config file for the OpenCASCADE package
# It defines the following variables
#  OpenCASCADE_INCLUDE_DIRS - include directories for OpenCASCADE
#  OpenCASCADE_LIBRARIES    - libraries to link against
#  OpenCASCADE_EXECUTABLE   - the bar executable

# Compute paths
set(OpenCASCADE_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(OpenCASCADE_INCLUDE_DIRS "${OpenCASCADE_CMAKE_DIR}/../inc")

# Our library dependencies (contains definitions for IMPORTED targets)
if (NOT TARGET foo AND NOT OpenCASCADE_BINARY_DIR)
    include("${OpenCASCADE_CMAKE_DIR}/OpenCASCADETargets.cmake")
endif ()

# These are IMPORTED targets created by OpenCASCADETargets.cmake
set(OpenCASCADE_LIBRARIES foo)
set(OpenCASCADE_EXECUTABLE bar)