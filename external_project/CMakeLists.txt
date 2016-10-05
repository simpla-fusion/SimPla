
SET(SAMRAI_DIR ./SAMRAI-master/)


INCLUDE_DIRECTORIES(${SAMRAI_DIR}/include)
INCLUDE_DIRECTORIES(${SAMRAI_DIR}/source)

FILE(GLOB_RECURSE SAMRAI_algs_SRC ${SAMRAI_DIR}/source/SAMRAI/algs/*.C)
FILE(GLOB_RECURSE SAMRAI_geom_SRC ${SAMRAI_DIR}/source/SAMRAI/geom/*.C)
FILE(GLOB_RECURSE SAMRAI_math_SRC ${SAMRAI_DIR}/source/SAMRAI/math/*.C)
FILE(GLOB_RECURSE SAMRAI_pdat_SRC ${SAMRAI_DIR}/source/SAMRAI/pdat/*.C)
FILE(GLOB_RECURSE SAMRAI_tbox_SRC ${SAMRAI_DIR}/source/SAMRAI/tbox/*.C)
FILE(GLOB_RECURSE SAMRAI_appu_SRC ${SAMRAI_DIR}/source/SAMRAI/appu/*.C)
FILE(GLOB_RECURSE SAMRAI_hier_SRC ${SAMRAI_DIR}/source/SAMRAI/hier/*.C)
FILE(GLOB_RECURSE SAMRAI_mesh_SRC ${SAMRAI_DIR}/source/SAMRAI/mesh/*.C)
FILE(GLOB_RECURSE SAMRAI_solv_SRC ${SAMRAI_DIR}/source/SAMRAI/solv/*.C)
FILE(GLOB_RECURSE SAMRAI_xfer_SRC ${SAMRAI_DIR}/source/SAMRAI/xfer/*.C)

ADD_LIBRARY(SAMRAI_algs ${SAMRAI_algs_SRC})
ADD_LIBRARY(SAMRAI_geom ${SAMRAI_geom_SRC})
ADD_LIBRARY(SAMRAI_math ${SAMRAI_math_SRC})
ADD_LIBRARY(SAMRAI_pdat ${SAMRAI_pdat_SRC})
ADD_LIBRARY(SAMRAI_tbox ${SAMRAI_tbox_SRC})
ADD_LIBRARY(SAMRAI_appu ${SAMRAI_appu_SRC})
ADD_LIBRARY(SAMRAI_hier ${SAMRAI_hier_SRC})
ADD_LIBRARY(SAMRAI_mesh ${SAMRAI_mesh_SRC})
ADD_LIBRARY(SAMRAI_solv ${SAMRAI_solv_SRC})
ADD_LIBRARY(SAMRAI_xfer ${SAMRAI_xfer_SRC})
FILE(GLOB_RECURSE SAMRAI_test_SRC ${SAMRAI_DIR}/source/test/testlib/*.C )
FILE(GLOB_RECURSE SAMRAI_test_SRC_f ${SAMRAI_DIR}/source/test/testlib/fortran/*.f)
ADD_LIBRARY(SAMRAI_test ${SAMRAI_test_SRC} ${SAMRAI_test_SRC_f})


ADD_LIBRARY(samrai_lib)
TARGET_LINK_LIBRARIES(samrai_lib
        SAMRAI_appu
        SAMRAI_algs
        SAMRAI_solv
        SAMRAI_geom
        SAMRAI_mesh
        SAMRAI_math
        SAMRAI_pdat
        SAMRAI_xfer
        SAMRAI_hier
        SAMRAI_tbox
        ${MPI_LIBRARIES} ${HDF5_LIBRARIES}
        )

#INCLUDE(ExternalProject)


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


#if(NOT MOAB_ROOT)
#ExternalProject_Add(moab
#    PREFIX third_part
#    URL file:///${PROJECT_SOURCE_DIR}/third_part/src/?????
#    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_part/?????
#    INSTALL_COMMAND "make" )
# ExternalProject_Get_Property(moab BINARY_DIR)
# SET(MOAB_INSTALL_DIR ${BINARY_DIR})
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