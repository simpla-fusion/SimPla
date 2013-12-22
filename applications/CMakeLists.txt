
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
              
ADD_EXECUTABLE(simpla  simpla.cpp )
TARGET_LINK_LIBRARIES(simpla physics ${LUA_LIBRARIES} io)

