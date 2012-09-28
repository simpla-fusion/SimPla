#- Find MKL (Intel Math Kernel Lib.)
#  This module will define the following variable:
# INPUT:
#  MKL_INTEGERS_LENGTH  32/64 
#  MKL_COMPUTATIONAL_LIBRARIES
# OUTPUT:
#  MKL_FOUND
#  MKL_ROOT_DIR
#  MKL_INCLUDE_DIR/MKL_C_INCLUDE_DIR
#  MKL_FORTRAN_INCLUDE_DIR
#  MKL_LIBRARY_DIR
#  MKL_STATIC_LIBRARIES
#  MKL_DYNAMIC_LIBRARIES

if(CMAKE_SYSTEM_PROCESSOR  STREQUAL "i686")
 set(_arch ia32) 
 set(_lp64_iface "")
elseif( CMAKE_SYSTEM_PROCESSOR  STREQUAL "x86_64")
 set(_arch "intel64")
 set(_lp64_iface "_lp64")
 if(MKL_INTEGERS_LENGTH EQUAL 64)
   set(_lp64_iface "_ilp64")
 endif()
endif()

if(NOT $ENV{MKL_ARCH} STREQUAL "" )
  set(_arch $ENV{MKL_ARCH})
endif()

if( MKL_INLCUDE_DIR AND MKL_SHARED_LIBRARIES AND MKL_STATIC_LIBRARIES)
#DO NOTHING
else()

    #solver lib
    set(_solver_lib "mkl_solver")
    
    if( _arch STREQUAL "em64t" OR  _arch STREQUAL "intel64" )
       set(_solver_lib "mkl_solver${_lp64_iface}")
    endif()
    #iface lib  
    #if(CMAKE_COMPILER_IS_GNUFORTRAN)
    #   set(_iface_fortran_lib "mkl_gf{_lp64_iface}")   
    #endif()
    set(_iface_lib "mkl_intel${_lp64_iface}")
      
    #thread lib
    if(${OPENMP_FOUND})
     if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"  )
       set(_thread_lib "mkl_gnu_thread")
       set(_omp_flag  "-lpthread"   )
     elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI" )
       set(_thread_lib "mkl_pgi_thread")
       set(_omp_flag  "-lpthread"   )
     elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" )
       set(_thread_lib "mkl_intel_thread")
       set(_omp_flag "-openmp"  "-lpthread"   )
     endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" )
    else()
     set(_thread_lib "mkl_sequential") 
     set(_solver_lib "${_solver_lib}_sequential")
    endif()
    
    #core lib  
    set(_core_lib "mkl_core")
      
    SET(MKL_LIBRARY_NAMES  ${_iface_lib} ${_thread_lib} ${_core_lib})
    
    foreach(LIB ${MKL_LIBRARY_NAMES})
    
    
     find_library("MKL_${LIB}_LIBRARY_STATIC" 
                NAME ${CMAKE_STATIC_LIBRARY_PREFIX}${LIB}${CMAKE_STATIC_LIBRARY_SUFFIX} 
                HINTS  /opt/intel/Compiler/*/* /opt/intel/composerxe/
                       $INTEL_HOME ENV INTEL_HOME
                PATH_SUFFIXES mkl/lib/${_arch}
                NO_DEFAULT_PATH
                )
     find_library("MKL_${LIB}_LIBRARY_SHARED" 
                NAME ${CMAKE_SHARED_LIBRARY_PREFIX}${LIB}${CMAKE_SHARED_LIBRARY_SUFFIX}
                HINTS  /opt/intel/Compiler/*/* /opt/intel/composerxe/
                       $INTEL_HOME ENV INTEL_HOME
                PATH_SUFFIXES mkl/lib/${_arch}
                NO_DEFAULT_PATH
                )
    endforeach()
    
    
#    find_library(MKL_SOLVER_LIBRARY_STATIC  
#                NAME ${CMAKE_STATIC_LIBRARY_PREFIX}${_solver_lib}${CMAKE_STATIC_LIBRARY_SUFFIX}
#                HINTS  /opt/intel/Compiler/*/*  /opt/intel/composerxe/
#                       $INTEL_HOME ENV INTEL_HOME
##                PATH_SUFFIXES mkl/lib/${_arch}
#                NO_DEFAULT_PATH
#                )
    find_path( MKL_LIBRARY_DIR 
            NAMES ${CMAKE_STATIC_LIBRARY_PREFIX}${_iface_lib}${CMAKE_STATIC_LIBRARY_SUFFIX}
            HINTS /opt/intel/Compiler/*/* /opt/intel/composerxe/
                  $INTEL_HOME ENV INTEL_HOME
            PATH_SUFFIXES mkl/lib/${_arch}
            NO_DEFAULT_PATH
            )            
                
    if(UNIX) 
      set(MKL_STATIC_LIBRARIES  -L${MKL_LIBRARY_DIR}
                                ${MKL_SOLVER_LIBRARY_STATIC} "-Wl,--start-group" 
                                ${MKL_${_iface_lib}_LIBRARY_STATIC} 
                                ${MKL_${_thread_lib}_LIBRARY_STATIC}
                                ${MKL_${_core_lib}_LIBRARY_STATIC}
                                 "-Wl,--end-group"  ${_omp_flag}         
                                )
       
      set(MKL_SHARED_LIBRARIES  -L${MKL_LIBRARY_DIR}
                                ${MKL_SOLVER_LIBRARY_STATIC} 
                                "-Wl,--start-group"
                                -l${_iface_lib} -l${_thread_lib} -l${_core_lib}
                                 "-Wl,--end-group"  ${_omp_flag}          
                                )
    
                             
    endif()
    
    find_path( MKL_C_INLCUDE_DIR 
            NAMES mkl.h mkl_vml.h mkl_vsl.h mkl_blas.h
            HINTS /opt/intel/Compiler/*/* /opt/intel/composerxe/
                  $INTEL_HOME ENV INTEL_HOME
            PATH_SUFFIXES mkl/include
            NO_DEFAULT_PATH
            )
    find_path( MKL_FORTRAN_INLCUDE_DIR 
            NAMES mkl.h mkl_vml.h mkl_vsl.h mkl_blas.h
            HINTS /opt/intel/Compiler/*/* /opt/intel/composerxe/
                  $INTEL_HOME ENV INTEL_HOME
            PATH_SUFFIXES mkl/include
            NO_DEFAULT_PATH
            )
            
    set(MKL_INLCUDE_DIR ${MKL_C_INLCUDE_DIR})
    
#    if(MKL_INLCUDE_DIR)
#     message(STATUS "Found MKL: static library:${MKL_STATIC_LIBRARIES} shared library:  ${MKL_SHARED_LIBRARIES}")
#    endif() 

    MARK_AS_ADVANCED(
      MKL_INLCUDE_DIR
      MKL_SHARED_LIBRARIES
      MKL_STATIC_LIBRARIES
    )
endif()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(IntelMKL  DEFAULT_MSG  MKL_STATIC_LIBRARIES)
#
#if(MKL_INLCUDE_DIR)
# message(STATUS "Found MKL_INLCUDE_DIR ${MKL_INLCUDE_DIR}")
# message(STATUS "Found MKL_LIBRARY_DIR ${MKL_LIBRARY_DIR}")
#else()
# message(STATUS "MKL Intel Math Kernel Lib. is no found.")
#endif() 

