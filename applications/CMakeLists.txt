
#ADD_SUBDIRECTORY( solver/electromagnetic )
#ADD_SUBDIRECTORY( fluid )
#ADD_SUBDIRECTORY( emfield )
#
#FIND_PACKAGE(Lua51)
#
#
#include_directories(BEFORE . 
#				${Boost_INCLUDE_DIRS}
#				${LUA_INCLUDE_DIR}
# )
# 
#ADD_EXECUTABLE(${PROJECT_NAME}  simpla.cpp ../src/engine/parseConfigFile.cpp)
#
#TARGET_LINK_LIBRARIES(${PROJECT_NAME}  engine io fluid emfield utilities pic                
#                 ${Boost_LIBRARIES}
#                 ${OpenMP_LIBRARIES}
#                 ${LUA_LIBRARIES}
#                 dl rt 
#                  )
#ADD_EXECUTABLE(WriteXDMF  WriteXDMF.cpp)

SET(MKL_LIBRARY_DIR   /opt/intel/composerxe/mkl/lib)
FIND_PACKAGE(IntelMKL)

ADD_EXECUTABLE(ggauge  ggauge.cpp)
TARGET_LINK_LIBRARIES(ggauge    engine physics io field_fun
              utilities  em ${MKL_STATIC_LIBRARIES} ${LUA_LIBRARIES})
              
ADD_EXECUTABLE(simpla  simpla.cpp)
TARGET_LINK_LIBRARIES(simpla physics ${LUA_LIBRARIES} ${Boost_LIBRARIES})

ADD_EXECUTABLE(2omega  2omega.cpp)
TARGET_LINK_LIBRARIES(2omega physics )