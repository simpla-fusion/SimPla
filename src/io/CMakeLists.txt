FILE(GLOB io_SRC ./*.cpp)
add_library(io ${io_SRC})

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