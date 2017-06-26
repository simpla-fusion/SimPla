cmake_minimum_required(VERSION 2.6)
project( test CXX )

# we need the M4 macro processor
find_program( M4_EXECUTABLE m4 DOC "The M4 macro processor" )
if( NOT M4_EXECUTABLE )
    message( SEND_ERROR "Failed to find the M4 macro processor." )
endif( NOT M4_EXECUTABLE )

# - Pass a list of files through the M4 macro processor
#
# ADD_M4_SOURCES( OUTVAR source1 ... sourceN )
#
#  OUTVAR  A list containing all the output file names, suitable
#          to be passed to add_executable or add_library.
#
# If the source files have a .m4 suffix it is stripped from the output
# file name. The output files are placed in the same relative location
# to CMAKE_CURRENT_BINARY_DIR as they are to CMAKE_CURRENT_SOURCE_DIR.
#
# Example:
#  add_m4_sources( SRCS src/test1.cxx.m4 src/test2.cxx.m4 )
#  add_executable( test ${SRCS} )
function( ADD_M4_SOURCES OUTVAR )
    set( outfiles )
    foreach( f ${ARGN} )
        # first we might need to make the input file absolute
        get_filename_component( f "${f}" ABSOLUTE )
        #if( NOT IS_ABSOLUTE "${f}" )
        #  set( f "${CMAKE_CURRENT_SOURCE_DIR}/${f}" )
        #endif( NOT IS_ABSOLUTE "${f}" )
        # get the relative path of the file to the current source dir
        file( RELATIVE_PATH rf "${CMAKE_CURRENT_SOURCE_DIR}" "${f}" )
        # strip the .m4 off the end if present and prepend the current  binary dir
        string( REGEX REPLACE "\\.m4$" ""  of "${CMAKE_CURRENT_BINARY_DIR}/${rf}" )
        # append the output file to the list of outputs
        list( APPEND outfiles "${of}" )
        # create the output directory if it doesn't exist
        get_filename_component( d "${of}" PATH )
        if( NOT IS_DIRECTORY "${d}" )
            file( MAKE_DIRECTORY "${d}" )
        endif( NOT IS_DIRECTORY "${d}" )
        # now add the custom command to generate the output file
        add_custom_command( OUTPUT "${of}"
                COMMAND ${M4_EXECUTABLE} ARGS -P -s "${f}" > "${of}"
                DEPENDS "${f}"
                )
    endforeach( f )
    # set the output list in the calling scope
    set( ${OUTVAR} ${outfiles} PARENT_SCOPE )
endfunction( ADD_M4_SOURCES )

add_m4_sources( SRCS src/test1.cxx.m4 src/test2.cxx.m4 )
add_executable( test ${SRCS} )