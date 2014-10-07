#- Find Silo 
# INPUT:
#  SILO_DIR
#  This module will define the following variable:
# OUTPUT:
#  SILO_INCLUDE_DIRS
#  SILO_LIBRARIES
include(SelectLibraryConfigurations)
include(FindPackageHandleStandardArgs)

if(SILO_INCLUDE_DIR )
#DO NOTHING
else()     
    find_path(SILO_INCLUDE_DIR
        NAMES silo.h
        HINTS ${SILO_DIR}/include
        NO_DEFAULT_PATH
       )
     find_library("SILO_LIBRARIES" 
                NAME ${CMAKE_STATIC_LIBRARY_PREFIX}silo${CMAKE_STATIC_LIBRARY_SUFFIX} 
                HINTS  ${SILO_DIR}/lib
                NO_DEFAULT_PATH
                )
      SET(SILO_FOUND true)    
endif()

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Silo  DEFAULT_MSG  SILO_INCLUDE_DIR )
