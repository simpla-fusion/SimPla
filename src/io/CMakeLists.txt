FILE(GLOB io_SRC ./*.cpp)
add_library(io ${io_SRC})
target_link_libraries(io ${HDF5_LIBRARIES})
#        /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5.a
#        /usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib/libhdf5_hl.a
#        /usr/lib/x86_64-linux-gnu/libsz.a
#        /usr/lib/x86_64-linux-gnu/libz.a
#        /usr/lib/x86_64-linux-gnu/libaec.a m dl


#find_package(VTK COMPONENTS
#        vtkCommonCore
#        vtkFiltersGeometry
#        vtkFiltersSources
#        vtkIOXML
#        vtkInteractionStyle
#        vtkRenderingQt
#        vtkTestingCore
#        vtksys
#        )
#include(${VTK_USE_FILE})
#
#if (vtkTestingCore_LOADED)
#    add_executable(MultiBlock MACOSX_BUNDLE MultiBlock.cxx)
#    target_link_libraries(MultiBlock ${VTK_LIBRARIES})
#endif ()