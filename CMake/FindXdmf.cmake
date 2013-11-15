#- Find libXdmf 
#  This module will define the following variable:
# OUTPUT:
#  XDMF_FOUND
#  XDMF_ROOT_DIR
#  XDMF_INCLUDE_DIR/XDMF_C_INCLUDE_DIR
#  XDMF_FORTRAN_INCLUDE_DIR
#  XDMF_LIBRARY_DIR
#  XDMF_STATIC_LIBRARIES
#  XDMF_DYNAMIC_LIBRARIES

 


if( XDMF_INLCUDE_DIR AND XDMF_SHARED_LIBRARIES AND XDMF_STATIC_LIBRARIES)
#DO NOTHING

else()

 find_library("XDMF_LIBRARY" 
          NAME $
                HINTS  /usr/lib
                
                NO_DEFAULT_PATH
                )
 find_library("XDMF_SHARED_LIBRARIES" 
                NAME ${CMAKE_SHARED_LIBRARY_PREFIX}Xdmf${CMAKE_SHARED_LIBRARY_SUFFIX}
                HINTS /usr/lib /usr/lib64                
                NO_DEFAULT_PATH
                
    
find_path( XDMF_INLCUDE_DIR 
            NAMES Xdmf.h            
            PATH_SUFFIXES /usr/include
            NO_DEFAULT_PATH
            )
    
#    if(XDMF_INLCUDE_DIR)
#     message(STATUS "Found XDMF: static library:${XDMF_STATIC_LIBRARIES} shared library:  ${XDMF_SHARED_LIBRARIES}")
#    endif() 

    MARK_AS_ADVANCED(
      XDMF_INLCUDE_DIR
      XDMF_SHARED_LIBRARIES
      XDMF_STATIC_LIBRARIES
    )
endif()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(XDMF  DEFAULT_MSG  XDMF_SHARED_LIBRARIES)
#
#if(XDMF_INLCUDE_DIR)
# message(STATUS "Found XDMF_INLCUDE_DIR ${XDMF_INLCUDE_DIR}")
# message(STATUS "Found XDMF_LIBRARY_DIR ${XDMF_LIBRARY_DIR}")
#else()
# message(STATUS "XDMF Intel Math Kernel Lib. is no found.")
#endif() 

