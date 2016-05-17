#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TKernel" for configuration "RelWithDebInfo"
set_property(TARGET TKernel APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TKernel PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "-lpthread;dl;-lm"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib//libTKernel.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libTKernel.so.10"
  )

list(APPEND _IMPORT_CHECK_TARGETS TKernel )
list(APPEND _IMPORT_CHECK_FILES_FOR_TKernel "${_IMPORT_PREFIX}/lib//libTKernel.so" )

# Import target "TKMath" for configuration "RelWithDebInfo"
set_property(TARGET TKMath APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TKMath PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "TKernel"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib//libTKMath.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libTKMath.so.10"
  )

list(APPEND _IMPORT_CHECK_TARGETS TKMath )
list(APPEND _IMPORT_CHECK_FILES_FOR_TKMath "${_IMPORT_PREFIX}/lib//libTKMath.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
