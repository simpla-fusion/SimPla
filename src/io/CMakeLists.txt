FILE(GLOB io_SRC ./*.cpp)
link_directories(${HDF5_LIBRARY_DIRS})
add_library(io ${io_SRC})
target_link_libraries(io ${HDF5_C_LIBRARIES} data_model)

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