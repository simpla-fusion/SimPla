#- Find NumPy 
# INPUT:
#  PYTHON_EXECUTABLE
#  This module will define the following variable:
# OUTPUT:
#  PYTHON_NUMPY_INCLUDE_DIRS

if(PYTHON_NUMPY_INCLUDE_DIRS )
#DO NOTHING
else()
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c 
    "
try: 
    import sys,numpy 
    sys.stdout.write(numpy.get_include()) 
except: 
    sys.stdout.write('PYTHON_NUMPY_INCLUDE_DIRS-NOTFOUND')
"
    OUTPUT_VARIABLE _RES
    )
    find_path(PYTHON_NUMPY_INCLUDE_DIRS
        NAMES numpy/arrayobject.h
        HINTS ${_RES}
        NO_DEFAULT_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_CMAKE_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_SYSTEM_PATH
       )
endif()


INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS( NumPy  DEFAULT_MSG PYTHON_NUMPY_INCLUDE_DIRS )
