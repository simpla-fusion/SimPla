INCLUDE(ExternalProject)





#ExternalProject_Add(trilinos
#    PREFIX third_part 
# #http://trilinos.sandia.gov/download/files/trilinos-11.10.2-Source.tar.bz2
#    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/trilinos-11.10.2-Source.tar.bz2
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/trilinos
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(trilinos SOURCE_DIR)
#ExternalProject_Get_Property(trilinos BINARY_DIR)


#if(NOT TBB_ROOT)
#ExternalProject_Add(tbb
#    PREFIX third_part 
#    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/tbb43_20141204oss_src.tgz
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/tbb43_20141204oss
#    INSTALL_COMMAND "make" )
# ExternalProject_Get_Property(tbb BINARY_DIR)
# SET(TBB_INSTALL_DIR ${BINARY_DIR})
#endif()





#SET(TBB_INSTALL_DIR  ${PROJECT_SOURCE_DIR}/third_part/tbb43_20140724oss)


#ExternalProject_Add(thrust
#    PREFIX third_part 
#    URL  http://thrust.googlecode.com/files/thrust-1.5.1.zip
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/thrust
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(thrust SOURCE_DIR)
#ExternalProject_Get_Property(thrust BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})

#ExternalProject_Add(pugixml
#    PREFIX third_part 
#    #    URL   https://pugixml.googlecode.com/files/pugixml-1.2.tar.gz
#    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/pugixml-1.2.tar.gz
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/pugixml
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(pugixml SOURCE_DIR)
#ExternalProject_Get_Property(pugixml BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})



#ExternalProject_Add(Eigen
#    PREFIX third_part 
#    URL  http://bitbucket.org/eigen/eigen/get/3.2.0.tar.bz2
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/Eigen
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(Eigen SOURCE_DIR)
#ExternalProject_Get_Property(Eigen BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})

#ExternalProject_Add(PHG
#    PREFIX third_part 
#    URL  http://lsec.cc.ac.cn/phg/download/phg-0.8.6-20120910.tar.bz2
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/Eigen
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(PHG SOURCE_DIR)
#ExternalProject_Get_Property(PHG BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/include )
#LINK_DIRECTORIES(${BINARY_DIR})

#ExternalProject_Add(Xdmf
#    PREFIX third_part
#    #GIT_REPOSITORY git://paraview.org/Xdmf.git
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/Xdmf
#    INSTALL_COMMAND "" )
#ExternalProject_Get_Property(Xdmf SOURCE_DIR)
#ExternalProject_Get_Property(Xdmf BINARY_DIR)
#INCLUDE_DIRECTORIES(${SOURCE_DIR}/libsrc ${BINARY_DIR}/libsrc )
#LINK_DIRECTORIES(${BINARY_DIR}/bin)
#SET(XDMF_NO_MPI ON)