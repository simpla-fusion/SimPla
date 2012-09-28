#- Find Python 
# INPUT:
#  PYTHONHOME  
#  PYTHONPATH
#  This module will define the following variable:
# OUTPUT:
#  PYTHON_VERSION
#  PYTHON_FOUND
#  PYTHON_INCLUDE_DIRS
#  PYTHON_LIBRARIES

if( PYTHON_EXECUTABLE )
#DO NOTHING
else()

    FOREACH(_CURRENT_VERSION 2.7 2.6 2.5 2.4)
      find_program(PYTHON_EXECUTABLE
		NAME python${_CURRENT_VERSION}${CMAKE_EXECUTABLE_SUFFIX}
		HINTS  $ENV{PYTHONHOME}/bin
 		)
      find_library(PYTHON_LIBRARIES 
                NAME ${CMAKE_SHARED_LIBRARY_PREFIX}python${_CURRENT_VERSION}${CMAKE_SHARED_LIBRARY_SUFFIX} 
                HINTS  $ENV{PYTHONHOME}/lib
                )
                
      find_path(PYTHON_INCLUDE_DIRS  
            NAMES Python.h
            HINTS  $ENV{PYTHONHOME}/include 
            PATH_SUFFIXES python${_CURRENT_VERSION}
            )

      if(  (${PYTHON_LIBRARIES} STREQUAL "PYTHON_LIBRARIES-NOTFOUND")  
          OR 
           ( ${PYTHON_INCLUDE_DIRS} STREQUAL "PYTHON_INCLUDE_DIRS-NOTFOUND") 
          OR 
           (${PYTHON_EXECUTABLE} STREQUAL "PYTHON_EXECUTABLE-NOTFOUND") 
           )
        #DO NOTHING
      else()
          set(PYTHON_VERSION "${_CURRENT_VERSION}")
          break()
      endif()    
    ENDFOREACH()
    
endif()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Python  DEFAULT_MSG PYTHON_EXECUTABLE )
